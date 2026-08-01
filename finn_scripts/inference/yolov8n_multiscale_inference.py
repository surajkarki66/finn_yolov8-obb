#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pyxrt

from qonnx.core.datatype import DataType
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)

from utils import (
    scale_boxes,
    xywhr2xyxyxyxy,
    non_max_suppression,
    make_anchors,
    dist2rbox,
    dfl_decode,
)

REG_MAX = 16

# Color palette (BGR for cv2)
HEX_PALETTE = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
    "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
    "520085", "CB38FF", "FF95C8", "FF37C7",
)


def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


COLORS = [hex2bgr(f"#{c}") for c in HEX_PALETTE]

io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [
        DataType["INT16"], DataType["INT13"],
        DataType["INT16"], DataType["INT13"],
        DataType["INT16"], DataType["INT14"],
    ],
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [
        (1, 52, 52, 1), (1, 52, 52, 65),
        (1, 26, 26, 1), (1, 26, 26, 65),
        (1, 13, 13, 1), (1, 13, 13, 65),
    ],
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [
        (1, 52, 52, 1, 1), (1, 52, 52, 65, 1),
        (1, 26, 26, 1, 1), (1, 26, 26, 65, 1),
        (1, 13, 13, 1, 1), (1, 13, 13, 65, 1),
    ],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [
        (1, 52, 52, 1, 2), (1, 52, 52, 65, 2),
        (1, 26, 26, 1, 2), (1, 26, 26, 65, 2),
        (1, 13, 13, 1, 2), (1, 13, 13, 65, 2),
    ],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0", "odma1", "odma2", "odma3", "odma4", "odma5"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 6,
}

# Branch layout, matching the order of odt/oshape_normal above:
#   0: angle_P3   1: box_P3
#   2: angle_P4   3: box_P4
#   4: angle_P5   5: box_P5
ANGLE_BRANCH_IDX = [0, 2, 4]
BOX_BRANCH_IDX = [1, 3, 5]


class FINNXRT:

    def __init__(
        self,
        xclbin,
        io_shape_dict,
        input_kernel_name="StreamingDataflowPartition_0",
        output_kernel_names=None,
        device_id=0,
    ):

        if output_kernel_names is None:
            output_kernel_names = [
                "StreamingDataflowPartition_2",
                "StreamingDataflowPartition_3",
                "StreamingDataflowPartition_4",
                "StreamingDataflowPartition_5",
                "StreamingDataflowPartition_6",
                "StreamingDataflowPartition_7",
            ]

        self.io_shape_dict = io_shape_dict
        self.output_kernel_names = output_kernel_names

        print("Opening device")
        self.device = pyxrt.device(device_id)

        print("Loading xclbin")
        xcl = pyxrt.xclbin(xclbin)
        self.uuid = self.device.load_xclbin(xcl)

        print("Creating kernels")
        self.idma0 = pyxrt.kernel(self.device, self.uuid, input_kernel_name)
        self.odma_kernels = [
            pyxrt.kernel(self.device, self.uuid, kernel_name)
            for kernel_name in self.output_kernel_names
        ]

        print("FINN accelerator ready")

    def alloc_bo(self, size):
        return pyxrt.bo(self.device, int(size), pyxrt.bo.normal, 0)

    def copy_to_bo(self, bo, data_bytes):
        mapped = bo.map()
        buf = np.frombuffer(mapped, dtype=np.uint8)
        buf[: data_bytes.size] = data_bytes.reshape(-1)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_bytes.nbytes, 0)

    def copy_from_bo(self, bo, size):
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)
        mapped = bo.map()
        return np.frombuffer(mapped, dtype=np.uint8, count=size).copy()

    def idt(self, ind=0):
        return self.io_shape_dict["idt"][ind]

    def odt(self, ind=0):
        return self.io_shape_dict["odt"][ind]

    def ishape_normal(self, ind=0):
        return self.io_shape_dict["ishape_normal"][ind]

    def oshape_normal(self, ind=0):
        return self.io_shape_dict["oshape_normal"][ind]

    def ishape_folded(self, ind=0):
        return self.io_shape_dict["ishape_folded"][ind]

    def oshape_folded(self, ind=0):
        return self.io_shape_dict["oshape_folded"][ind]

    def ishape_packed(self, ind=0):
        return self.io_shape_dict["ishape_packed"][ind]

    def oshape_packed(self, ind=0):
        return self.io_shape_dict["oshape_packed"][ind]

    def fold_input(self, ibuf_normal, ind=0):
        assert ibuf_normal.shape == self.ishape_normal(ind), (
            f"Expected input shape {self.ishape_normal(ind)}, got {ibuf_normal.shape}"
        )
        return ibuf_normal.reshape(self.ishape_folded(ind))

    def pack_input(self, ibuf_folded, ind=0):
        return finnpy_to_packed_bytearray(
            ibuf_folded,
            self.idt(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )

    def unpack_output(self, obuf_packed, ind=0):
        return packed_bytearray_to_finnpy(
            obuf_packed,
            self.odt(ind),
            self.oshape_folded(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )

    def unfold_output(self, obuf_folded, ind=0):
        return obuf_folded.reshape(self.oshape_normal(ind))

    def execute(self, image):
        ibuf_folded = self.fold_input(image.astype(np.uint8), ind=0)
        ibuf_packed = self.pack_input(ibuf_folded, ind=0)
        ibuf_packed = np.ascontiguousarray(ibuf_packed, dtype=np.uint8)

        ibuf_bo = self.alloc_bo(ibuf_packed.size)
        self.copy_to_bo(ibuf_bo, ibuf_packed)

        out_packed_sizes = [
            int(np.prod(self.oshape_packed(i)))
            for i in range(self.io_shape_dict["num_outputs"])
        ]
        obuf_bos = [self.alloc_bo(size) for size in out_packed_sizes]

        print("Starting output DMA(s)")
        output_runs = [kernel(bo, 1) for kernel, bo in zip(self.odma_kernels, obuf_bos)]

        print("Starting input DMA")
        input_run = self.idma0(ibuf_bo, 1)

        print("Waiting")
        input_run.wait()
        for run in output_runs:
            run.wait()

        print("Inference done")

        outputs = []
        for idx, (bo, out_packed_size) in enumerate(zip(obuf_bos, out_packed_sizes)):
            raw = self.copy_from_bo(bo, out_packed_size)
            packed = raw.reshape(self.oshape_packed(idx))
            folded = self.unpack_output(packed, ind=idx)
            normal = self.unfold_output(folded, ind=idx)
            outputs.append(normal)

        return outputs


def load_scales(scale_dir):
    muls = [np.load(f"{scale_dir}/mul_{i}.npy") for i in range(6)]
    adds = [np.load(f"{scale_dir}/add_{i}.npy") for i in range(6)]
    return muls, adds


def prepare_outputs_for_decode(outputs, muls, adds):
    """NHWC accelerator outputs -> NCHW float outputs, with affine dequant applied.

    Returns (box_outputs, angle_outputs), each a list of 3 NCHW arrays,
    ordered P3, P4, P5.
    """
    dequant = []
    for output, mul, add in zip(outputs, muls, adds):
        out = output.transpose(0, 3, 1, 2).astype(np.float64)  # NHWC -> NCHW
        out = out * mul + add
        dequant.append(out)

    angle_outputs = [dequant[i] for i in ANGLE_BRANCH_IDX]
    box_outputs = [dequant[i] for i in BOX_BRANCH_IDX]
    return box_outputs, angle_outputs


# -----------------------------------------------------------------
# YOLOv8-OBB postprocessing
# -----------------------------------------------------------------
def outputs_to_prediction(box_outputs, angle_outputs, imgsz, nc, reg_max=REG_MAX):
    """Convert 6 dequantized accelerator outputs to prediction tensor
    [1, 4+nc+1, total_anchors] for NMS.

    Steps:
    1) make_anchors from feats + strides  2) concat box outputs, split (box_raw, cls)
    3) DFL decode box_raw -> pred_dist   4) angle: concat raw per-scale, (sigmoid-0.25)*pi
    5) dist2rbox(pred_dist, angle, anchors, dim=1) * strides -> dbox
    6) cat(dbox, cls.sigmoid(), angle) -> [cx,cy,w,h, cls..., angle] for NMS.
    """
    feats = box_outputs
    strides = np.array([imgsz / feats[i].shape[2] for i in range(len(feats))], dtype=np.float64)
    anchor_points, stride_tensor = make_anchors(feats, strides)
    anchor_points = anchor_points[None, :, :]      # (1, N, 2)
    stride_tensor = stride_tensor[None, :, 0]      # (1, N)

    shape = feats[0].shape
    no = shape[1]
    assert no == nc + reg_max * 4
    x_cat = np.concatenate([xi.reshape(shape[0], no, -1) for xi in feats], axis=2)

    box_raw = x_cat[:, : reg_max * 4, :]
    cls = x_cat[:, reg_max * 4:, :]

    pred_dist = dfl_decode(box_raw, reg_max)
    cls = np.clip(cls, -20.0, 20.0)

    angle_list = [a.reshape(a.shape[0], -1) for a in angle_outputs]
    angle = np.concatenate(angle_list, axis=1)[:, None, :]  # (1, 1, N)

    if angle.min() >= -0.01 and angle.max() <= 1.01:
        angle = (angle - 0.25) * math.pi
    else:
        angle_sig = 1.0 / (1.0 + np.exp(-np.clip(angle, -20.0, 20.0)))
        angle = (angle_sig - 0.25) * math.pi

    dbox = dist2rbox(pred_dist, angle, anchor_points, dim=1) * stride_tensor[:, None, :]
    cls_sig = 1.0 / (1.0 + np.exp(-cls))
    y = np.concatenate([dbox, cls_sig], axis=1)
    pred = np.concatenate([y, angle], axis=1)
    return pred


def get_names(data_path: str, nc: int) -> dict:
    """Load class names from data.yaml if path given; else 0,1,...,nc-1."""
    if not data_path or not Path(data_path).exists():
        return {i: str(i) for i in range(nc)}
    import yaml
    with open(data_path) as f:
        d = yaml.safe_load(f) or {}
    names = d.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return names if names else {i: str(i) for i in range(nc)}


def draw_obb_label(img, box_xyxyxyxy, label, color, line_thickness=2, font_scale=0.5, font_thickness=1):
    """Draw rotated box (4 corners) and label with filled background."""
    box = np.asarray(box_xyxyxyxy, dtype=np.int32)
    cv2.polylines(img, [box], True, color, line_thickness, cv2.LINE_AA)
    if label:
        p1 = (int(box[0][0]), int(box[0][1]))
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        outside = p1[1] - h >= 3
        p2 = (p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3)
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img, label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )


def preprocess(img: np.ndarray, imgsz: int):
    """Letterbox-resize BGR image to (imgsz, imgsz) and produce the
    accelerator's expected uint8 NHWC input, plus scale/pad info for
    rescaling boxes back to the original image later.
    """
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dh = (imgsz - new_h) / 2
    dw = (imgsz - new_w) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_in = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR -> RGB, HWC uint8, NHWC batch dim -- accelerator expects raw uint8 (no /255 normalization)
    img_in = img_in[:, :, ::-1]
    img_in = np.ascontiguousarray(img_in).astype(np.uint8)
    driver_in = np.expand_dims(img_in, 0)

    return driver_in, r, (dw, dh)


def parse_args():
    p = argparse.ArgumentParser(description="pyxrt FINN inference for YOLOv8-OBB (pure NumPy)")
    p.add_argument("--xclbin", default="finn-accel.xclbin", help="Path to the compiled xclbin")
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument("--scale-dir", type=str, default=".", help="Directory with mul_0..5.npy / add_0..5.npy")
    p.add_argument("--data", type=str, default="", help="Data YAML for class names (optional)")
    p.add_argument("--nc", type=int, default=1, help="Number of classes")
    p.add_argument("--imgsz", type=int, default=416, help="Input size (H and W); must match accelerator build")
    p.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--reg-max", type=int, default=REG_MAX, help="DFL reg_max (default 16)")
    p.add_argument("--save", action="store_true", help="Save annotated images")
    p.add_argument("--show", action="store_true", help="Show result in window")
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="predict_fpga")
    p.add_argument("--device-id", type=int, default=0, help="Alveo device index")
    p.add_argument(
        "--input-kernel",
        default="StreamingDataflowPartition_0",
        help="Input kernel name in the xclbin",
    )
    p.add_argument(
        "--output-kernels",
        nargs="*",
        default=None,
        help="Output kernel names in the xclbin (6 required, in odma0..odma5 order)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    finn = FINNXRT(
        args.xclbin,
        io_shape_dict=io_shape_dict,
        input_kernel_name=args.input_kernel,
        output_kernel_names=args.output_kernels,
        device_id=args.device_id,
    )

    nc = args.nc
    reg_max = args.reg_max
    names = get_names(args.data, nc)
    imgsz = args.imgsz

    print("Loading scales")
    muls, adds = load_scales(args.scale_dir)

    source = Path(args.source)
    if source.is_dir():
        files = list(source.rglob("*.*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    else:
        files = [source] if source.exists() else []

    if not files:
        raise FileNotFoundError(f"No images found at {args.source}")

    save_dir = Path(args.project) / args.name
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_dir}")

    for path in files:
        print(f"\nProcessing {path}")
        img = cv2.imread(str(path))
        if img is None:
            continue
        h0, w0 = img.shape[:2]

        driver_in, r, (dw, dh) = preprocess(img, imgsz)

        # -----------------------------------------------------
        # Run inference on FPGA
        # -----------------------------------------------------
        raw_outputs = finn.execute(driver_in)

        # Apply mul+add dequant, split into box/angle branches, NHWC -> NCHW
        box_outputs, angle_outputs = prepare_outputs_for_decode(raw_outputs, muls, adds)

        # -----------------------------------------------------
        # Postprocessing: DFL + angle decode -> rotated NMS -> rescale
        # -----------------------------------------------------
        pred = outputs_to_prediction(box_outputs, angle_outputs, imgsz=imgsz, nc=nc, reg_max=reg_max)
        pred = non_max_suppression(pred, conf_thres=args.conf, iou_thres=args.iou, nc=nc)[0]

        if len(pred):
            pred[:, :4] = scale_boxes(
                (imgsz, imgsz), pred[:, :4], (h0, w0),
                ratio_pad=((r,), (dw, dh)), xywh=True,
            )
            # column order from non_max_suppression: [x, y, w, h, conf, cls, angle]
            xywh = pred[:, :4]
            angle_col = pred[:, -1:]
            conf_cls = pred[:, 4:6]
            obb = np.concatenate([xywh, angle_col, conf_cls], axis=-1)

            for row in obb:
                *xywhr, conf, cls = row.tolist()
                corners = xywhr2xyxyxyxy(np.array([xywhr]))
                box_xyxyxyxy = corners[0]
                c = int(cls)
                name = names.get(c, str(c))
                label = f"{name} {conf:.2f}"
                color = COLORS[c % len(COLORS)]
                draw_obb_label(img, box_xyxyxyxy, label, color)
            print(f"Found {len(pred)} detection(s)")
        else:
            print("No detections")

        if args.save:
            out_path = save_dir / path.name
            cv2.imwrite(str(out_path), img)
            print(f"Saved annotated image to {out_path}")

        if args.show:
            cv2.imshow(str(path), img)
            cv2.waitKey(0 if len(files) == 1 else 1)

    if args.show:
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()