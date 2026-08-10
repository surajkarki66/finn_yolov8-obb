import argparse
import time
import cv2
import numpy as np
import pyxrt

from qonnx.core.datatype import DataType
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)

from utils import (
    letterbox,
    visualize_boxes,
)

io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT21"], DataType["INT21"], DataType["INT22"]],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [
        (1, 52, 52, 65),
        (1, 26, 26, 65),
        (1, 13, 13, 65),
    ],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [
        (1, 52, 52, 65, 1),
        (1, 26, 26, 65, 1),
        (1, 13, 13, 65, 1),
    ],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [
        (1, 52, 52, 65, 3),
        (1, 26, 26, 65, 3),
        (1, 13, 13, 65, 3),
    ],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0", "odma1", "odma2"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 3,
}


class FINNXRT:
    """
    Hardened version of the original FINNXRT:
      - usable as a context manager (`with FINNXRT(...) as finn:`) so
        device/kernel handles are always released, even on exception
      - BOs allocated once in __init__ and reused across execute() calls
        instead of allocated fresh per image (lower per-call overhead)
      - bounded wait on each DMA run (default 15s) instead of an
        unbounded run.wait(), so a stalled/deadlocked CU (e.g. from a
        thermal/power clock-stop event -- check `dmesg` for "Critical
        temperature or power event" / "Card requires pci hot reset")
        raises a clear TimeoutError instead of hanging the process
        forever and leaving handles open for the next invocation to
        inherit a wedged device.
    """

    def __init__(
        self,
        xclbin,
        io_shape_dict,
        input_kernel_name="StreamingDataflowPartition_0",
        output_kernel_names=None,
        device_id=0,
        run_timeout_s=15.0,
    ):

        if output_kernel_names is None:
            # xclbinutil reports odma0/odma1/odma2 as StreamingDataflowPartition_2/3/4.
            output_kernel_names = [
                "StreamingDataflowPartition_2",
                "StreamingDataflowPartition_3",
                "StreamingDataflowPartition_4",
            ]

        self.io_shape_dict = io_shape_dict
        self.output_kernel_names = output_kernel_names
        self.last_fpga_latency_ms = None
        self.run_timeout_s = run_timeout_s

        self.device = None
        self.idma0 = None
        self.odma_kernels = []
        self._ibuf_bo = None
        self._obuf_bos = []

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

        # Pre-allocate BOs once; reused across every execute() call.
        ibuf_packed_size = int(np.prod(self.ishape_packed(0)))
        self._ibuf_bo = self.alloc_bo(ibuf_packed_size, self.idma0, 0)

        out_packed_sizes = [
            int(np.prod(self.oshape_packed(i)))
            for i in range(self.io_shape_dict["num_outputs"])
        ]
        self._obuf_bos = [
            self.alloc_bo(size, self.odma_kernels[i], 0)
            for i, size in enumerate(out_packed_sizes)
        ]
        self._out_packed_sizes = out_packed_sizes

        print("FINN accelerator ready")

    # ---- context manager: guarantees cleanup on exit or exception ----
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # never swallow exceptions

    def close(self):
        """Best-effort release of native handles. Safe to call multiple times."""
        self._ibuf_bo = None
        self._obuf_bos = []
        self.odma_kernels = []
        self.idma0 = None
        self.device = None
        print("FINNXRT: device handles released")

    def alloc_bo(self, size, kernel, argno=0):
        return pyxrt.bo(self.device, int(size), pyxrt.bo.normal, kernel.group_id(argno))

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

    # ---- bounded wait, tolerant of pyxrt API differences --------------
    def _wait_bounded(self, run, label, timeout_s=None):
        timeout_s = timeout_s or self.run_timeout_s
        t0 = time.monotonic()
        try:
            # Newer pyxrt: run.wait(timeout_ms) returns ert_cmd_state
            return run.wait(int(timeout_s * 1000))
        except TypeError:
            # Older pyxrt: run.wait() takes no args and blocks indefinitely.
            # Poll run.state() instead so we can bail out after timeout_s.
            while time.monotonic() - t0 < timeout_s:
                try:
                    state = run.state()
                    s = str(state).upper()
                    if "RUNNING" not in s and "QUEUED" not in s and "NEW" not in s:
                        return state
                except Exception:
                    pass
                time.sleep(0.05)
            raise TimeoutError(
                f"{label} did not complete within {timeout_s}s -- likely a stalled "
                f"CU. Check `dmesg | tail -50` for 'Critical temperature or power "
                f"event' / 'Card requires pci hot reset'. If present, this needs "
                f"an admin-level PCIe hot reset, not a script retry."
            )

    def execute(self, image):
        ibuf_folded = self.fold_input(image.astype(np.uint8), ind=0)
        ibuf_packed = self.pack_input(ibuf_folded, ind=0)
        ibuf_packed = np.ascontiguousarray(ibuf_packed, dtype=np.uint8)

        try:
            self.copy_to_bo(self._ibuf_bo, ibuf_packed)

            print("Starting output DMA(s)")
            fpga_t0 = time.perf_counter()
            output_runs = [
                kernel(bo, 1) for kernel, bo in zip(self.odma_kernels, self._obuf_bos)
            ]

            print("Starting input DMA")
            input_run = self.idma0(self._ibuf_bo, 1)

            print("Waiting")
            self._wait_bounded(input_run, "input DMA")
            for i, run in enumerate(output_runs):
                self._wait_bounded(run, f"output DMA {i}")

            fpga_t1 = time.perf_counter()
            self.last_fpga_latency_ms = (fpga_t1 - fpga_t0) * 1000.0

            print("Inference done")
            print(f"FPGA latency: {self.last_fpga_latency_ms:.3f} ms")

        except TimeoutError as e:
            print(f"ERROR: {e}")
            print(
                "This process will exit without touching the device further. "
                "Retrying immediately will likely just fail to (re)program the "
                "xclbin if the card is in a thermal/power protection state."
            )
            raise
        except Exception:
            print("ERROR during execute() -- releasing device handles before re-raising")
            self.close()
            raise

        outputs = []
        for idx, (bo, out_packed_size) in enumerate(zip(self._obuf_bos, self._out_packed_sizes)):
            raw = self.copy_from_bo(bo, out_packed_size)
            packed = raw.reshape(self.oshape_packed(idx))
            folded = self.unpack_output(packed, ind=idx)
            normal = self.unfold_output(folded, ind=idx)
            outputs.append(normal)

        return outputs


def load_scales(scale_dir):
    muls = [
        np.load(f"{scale_dir}/mul_0.npy"),
        np.load(f"{scale_dir}/mul_1.npy"),
        np.load(f"{scale_dir}/mul_2.npy"),
    ]
    adds = [
        np.load(f"{scale_dir}/add_0.npy"),
        np.load(f"{scale_dir}/add_1.npy"),
        np.load(f"{scale_dir}/add_2.npy"),
    ]
    return muls, adds


def prepare_outputs_for_decode(outputs, muls, adds):
    """NHWC accelerator outputs -> NCHW float outputs, with affine dequant applied."""
    prepared = []
    for output, mul, add in zip(outputs, muls, adds):
        output = output.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC -> NCHW
        output = output * mul + add
        prepared.append(output)
    return prepared


# -----------------------------------------------------------------
# YOLOv8 postprocessing
# -----------------------------------------------------------------
def make_anchors(output_shapes, strides, grid_cell_offset=0.5):
    """Generate anchors from features"""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = output_shapes[i]  # NCHW format
        sx = np.arange(start=grid_cell_offset, stop=w, step=1)
        sy = np.arange(start=grid_cell_offset, stop=h, step=1)
        sx, sy = np.meshgrid(sx, sy)  # Note: order matters! sx, sy not sy, sx
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append([stride] * (h * w))

    # Transpose to match expected format [1, 2, num_anchors]
    anchor_points = np.expand_dims(np.concatenate(anchor_points).transpose(1, 0), 0)
    strides_tensor = np.concatenate(stride_tensor)
    return anchor_points, strides_tensor


def yolov8_postprocess(outs, batch_size, anchor_points, strides, nc=1, dfl_ch=16):
    """
    YOLOv8 postprocessing (NCHW format).
    """
    # DFL integration weights
    dfl_integration_weights = np.arange(dfl_ch).reshape(1, -1, 1, 1)

    # Concatenate outputs: each is [batch, channels, H, W]
    # Reshape to [batch, channels, H*W] then concatenate
    x_cat = np.concatenate([out.reshape(batch_size, nc + dfl_ch * 4, -1) for out in outs], 2)

    # Split into boxes and classes
    boxes_classes = np.split(x_cat, [dfl_ch * 4], axis=1)
    boxes, classes = boxes_classes

    # Apply sigmoid to classes
    classes = 1 / (1 + np.exp(-np.clip(classes, -50, 50)))

    # DFL - apply softmax via exp normalization
    boxes = boxes.reshape(batch_size, 4, dfl_ch, -1).transpose(0, 2, 1, 3)
    exp_boxes = np.exp(np.clip(boxes, -50, 50))
    boxes = exp_boxes / np.sum(exp_boxes, axis=1, keepdims=True)
    boxes *= dfl_integration_weights
    boxes = np.sum(boxes, 1)  # [batch, 4, num_anchors]

    # Decode boxes from ltrb to xyxy
    left_top_right_bottom = np.split(boxes, 2, axis=1)
    lt = left_top_right_bottom[0]
    rb = left_top_right_bottom[1]
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    boxes = np.concatenate((x1y1, x2y2), 1) * strides

    # Combine predictions
    pred = np.concatenate((boxes, classes), 1)

    return pred


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):

    def nms(boxes, scores, overlap_threshold=0.5):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        index_array = scores.argsort()[::-1]
        keep = []
        while index_array.size > 0:
            keep.append(index_array[0])
            x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
            y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
            x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
            y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

            w = np.maximum(0.0, x2_ - x1_ + 1)
            h = np.maximum(0.0, y2_ - y1_ + 1)
            inter = w * h
            overlap = inter / (areas[index_array[0]] + areas[index_array[1:]] - inter)

            inds = np.where(overlap <= overlap_threshold)[0]
            index_array = index_array[inds + 1]
        return keep

    bs = prediction.shape[0]
    xc = np.max(prediction[:, 4:], axis=1) > conf_thres

    prediction = prediction.transpose(0, 2, 1)
    output = [np.zeros((0, 6))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        box_cls = np.split(x, [4], axis=1)
        box = box_cls[0]
        cls = box_cls[1]

        j = cls.argmax(1, keepdims=True)
        conf = np.take_along_axis(x[:, 4:], j, axis=1)
        x = np.concatenate((box, conf, j), 1)

        n = x.shape[0]
        if not n:
            continue

        scores = x[:, 4]
        boxes = x[:, :4]
        i = nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output[0] if bs == 1 else output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) - with letterbox support."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip coordinates
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])

    return coords


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Execute YOLOv8n FINN accelerator on a single test image")
    parser.add_argument("--xclbin", default="finn-accel.xclbin", help="Path to the compiled xclbin")
    parser.add_argument("--image", default="data/3.jpg", help="Input image path")
    parser.add_argument(
        "--scale-dir", default=".", help="Directory containing mul_0/1/2.npy and add_0/1/2.npy"
    )
    parser.add_argument("--output-path", default="annotated_test.jpg", help="Path for the annotated output image")
    parser.add_argument("--device-id", type=int, default=0, help="Alveo device index")
    parser.add_argument(
        "--input-kernel",
        default="StreamingDataflowPartition_0",
        help="Input kernel name in the xclbin",
    )
    parser.add_argument(
        "--output-kernels",
        nargs="*",
        default=[
            "StreamingDataflowPartition_2",
            "StreamingDataflowPartition_3",
            "StreamingDataflowPartition_4",
        ],
        help="Output kernel names in the xclbin",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for each DMA run before raising a TimeoutError (default 15s)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------
    # Model / postprocessing configuration
    # -----------------------------------------------------

    names = ["person"]
    nc = 1
    dfl_ch = 16
    strides = [8, 16, 32]

    conf_thres = 0.25
    iou_thres = 0.45
    img_size = (416, 416)

    print("Loading scales")
    muls, adds = load_scales(args.scale_dir)

    # -----------------------------------------------------
    # Load input image
    # -----------------------------------------------------

    print(f"Loading {args.image}")
    img_org = cv2.imread(args.image)
    if img_org is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    original_shape = img_org.shape[:2]  # (height, width)

    # `with` guarantees device/kernel handles are released even if execute()
    # raises (e.g. a TimeoutError from a stalled/thermally-tripped CU).
    with FINNXRT(
        args.xclbin,
        io_shape_dict=io_shape_dict,
        input_kernel_name=args.input_kernel,
        output_kernel_names=args.output_kernels,
        device_id=args.device_id,
        run_timeout_s=args.run_timeout,
    ) as finn:

        # -----------------------------------------------------
        # End-to-end timer starts here (preprocess -> FPGA -> decode -> NMS)
        # -----------------------------------------------------
        e2e_t0 = time.perf_counter()

        img = img_org.copy()
        img, _, _ = letterbox(img, img_size, auto=False)
        img = img[:, :, ::-1]
        img = img.astype(np.uint8)
        driver_in = np.expand_dims(img, 0)

        # Gain / padding for later coordinate scaling (same convention as
        # the reference yolov8_quant_inference notebook)
        gain = min(img_size[0] / original_shape[0], img_size[1] / original_shape[1])
        pad = (
            (img_size[1] - original_shape[1] * gain) / 2,  # x padding
            (img_size[0] - original_shape[0] * gain) / 2,  # y padding
        )

        # -----------------------------------------------------
        # Run inference on FPGA (FPGA-only latency measured inside finn.execute)
        # -----------------------------------------------------

        print("Running inference")
        outputs = finn.execute(driver_in)

        # Apply mul+add dequant and convert NHWC -> NCHW for YOLOv8 decode
        outputs = prepare_outputs_for_decode(outputs, muls, adds)
        output_shapes = [out.shape for out in outputs]  # NCHW shapes for anchor generation

        # -----------------------------------------------------
        # Postprocessing: DFL decode + NMS
        # -----------------------------------------------------

        anchor_points, strides_tensor = make_anchors(output_shapes, strides)

        pred = yolov8_postprocess(outputs, batch_size=1, anchor_points=anchor_points,
                                   strides=strides_tensor, nc=nc, dfl_ch=dfl_ch)

        detections = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=300)

        # -----------------------------------------------------
        # End-to-end timer stops here: detection bounding boxes are ready
        # -----------------------------------------------------
        e2e_t1 = time.perf_counter()
        end2end_latency_ms = (e2e_t1 - e2e_t0) * 1000.0

        print("\n=== Latency ===")
        print(f"FPGA-only latency:  {finn.last_fpga_latency_ms:.3f} ms")
        print(f"End-to-end latency: {end2end_latency_ms:.3f} ms  "
              f"(preprocess + FPGA + dequant + decode + NMS)")

        boxes_detected, class_names_detected, probs_detected = [], [], []

        if len(detections):
            detections[:, :4] = scale_coords(
                img_size, detections[:, :4], original_shape, ratio_pad=(gain, pad)
            )

            for c in np.unique(detections[:, -1]):
                n = (detections[:, -1] == c).sum()
                print(f"{n} {names[int(c)]}")

            for *xyxy, conf, cls in detections:
                c = int(cls)
                boxes_detected.append(xyxy)
                class_names_detected.append(names[c])
                probs_detected.append(conf)
        else:
            print("No detections")

        # -----------------------------------------------------
        # Draw boxes and save annotated image
        # -----------------------------------------------------

        image_boxes = visualize_boxes(
            img_org, boxes_detected, class_names_detected, probs_detected
        )

        cv2.imwrite(args.output_path, image_boxes)
        print(f"Saved annotated image to {args.output_path}")