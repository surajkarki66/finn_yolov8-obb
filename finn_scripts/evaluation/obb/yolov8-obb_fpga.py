import argparse
import glob
import os
import time

import cv2
import numpy as np
import pyxrt

from finn_scripts.evaluation.obb.qonnx.core.datatype import DataType
from finn_scripts.evaluation.obb.finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)

from utils import letterbox

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt": [DataType['UINT8']],
    "odt": [
        DataType['INT16'], DataType['INT13'],
        DataType['INT16'], DataType['INT13'],
        DataType['INT16'], DataType['INT14'],
    ],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [
        (1, 52, 52, 1), (1, 52, 52, 65),
        (1, 26, 26, 1), (1, 26, 26, 65),
        (1, 13, 13, 1), (1, 13, 13, 65),
    ],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
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
    "input_dma_name": ['idma0'],
    "output_dma_name": ['odma0', 'odma1', 'odma2', 'odma3', 'odma4', 'odma5'],
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

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


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

        # --- Allocate all buffer objects ONCE and reuse them for every image ---
        print("Pre-allocating buffer objects")
        ishape_packed_size = int(np.prod(self.ishape_packed(0)))
        self.ibuf_bo = self.alloc_bo(ishape_packed_size, self.idma0, 0)

        self.out_packed_sizes = [
            int(np.prod(self.oshape_packed(i)))
            for i in range(self.io_shape_dict["num_outputs"])
        ]
        self.obuf_bos = [
            self.alloc_bo(size, self.odma_kernels[i], 0)
            for i, size in enumerate(self.out_packed_sizes)
        ]

        print("FINN accelerator ready")

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

    def execute(self, image, verbose=True, timing=None):
        def log(msg):
            if verbose:
                print(msg)

        t0 = time.time()
        ibuf_folded = self.fold_input(image.astype(np.uint8), ind=0)
        ibuf_packed = self.pack_input(ibuf_folded, ind=0)
        ibuf_packed = np.ascontiguousarray(ibuf_packed, dtype=np.uint8)
        t_prep = time.time()

        # Reuse the pre-allocated input BO instead of allocating a new one
        self.copy_to_bo(self.ibuf_bo, ibuf_packed)
        t_alloc = time.time()

        log("Starting output DMA(s)")
        output_runs = [
            kernel(bo, 1) for kernel, bo in zip(self.odma_kernels, self.obuf_bos)
        ]

        log("Starting input DMA")
        input_run = self.idma0(self.ibuf_bo, 1)

        log("Waiting")
        input_run.wait()
        for run in output_runs:
            run.wait()
        t_exec = time.time()

        log("Inference done")

        outputs = []
        for idx, (bo, out_packed_size) in enumerate(zip(self.obuf_bos, self.out_packed_sizes)):
            raw = self.copy_from_bo(bo, out_packed_size)
            packed = raw.reshape(self.oshape_packed(idx))
            folded = self.unpack_output(packed, ind=idx)
            normal = self.unfold_output(folded, ind=idx)
            outputs.append(normal)
        t_post = time.time()

        if timing is not None:
            timing["prep_s"] = t_prep - t0
            timing["alloc_s"] = t_alloc - t_prep
            timing["exec_s"] = t_exec - t_alloc
            timing["postproc_s"] = t_post - t_exec
            timing["total_s"] = t_post - t0

        return outputs


def list_images(data_dir):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(data_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def preprocess_image(image_path, img_size):
    img_org = cv2.imread(image_path)
    if img_org is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    img, ratio, (dw, dh) = letterbox(img_org.copy(), img_size, auto=False)
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.uint8)
    driver_in = np.expand_dims(img, 0)
    return driver_in, img_org, ratio, (dw, dh)


def save_raw_outputs(
    outputs, io_shape_dict, image_path, out_path, orig_shape, ratio, pad
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_dict = {}
    for i, out in enumerate(outputs):
        save_dict[f"output_{i}"] = out  # raw int values, shape = oshape_normal[i]

    save_dict["image_path"] = np.array(str(image_path))
    save_dict["odt"] = np.array([str(dt) for dt in io_shape_dict["odt"]])
    save_dict["oshape_normal"] = np.array(io_shape_dict["oshape_normal"], dtype=object)

    # Letterbox metadata, needed later to rescale predicted boxes back to
    # the original image coordinates.
    save_dict["orig_shape"] = np.array(orig_shape)  # (H, W, C) of original image
    save_dict["ratio"] = np.array(ratio)             # (rw, rh) or scalar r
    save_dict["pad"] = np.array(pad)                 # (dw, dh)

    np.savez(out_path, **save_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FINN/XRT accelerator on all images in a folder and save raw output tensors (YOLOv8-OBB, 6 output branches: angle/box per scale)"
    )
    parser.add_argument("--xclbin", default="finn-accel.xclbin", help="Path to the compiled xclbin")
    parser.add_argument("--data-dir", default="data", help="Folder containing input images")
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
            "StreamingDataflowPartition_5",
            "StreamingDataflowPartition_6",
            "StreamingDataflowPartition_7",
        ],
        help="Output kernel names in the xclbin (order must match io_shape_dict output order: "
             "[angle_P3, box_P3, angle_P4, box_P4, angle_P5, box_P5])",
    )
    parser.add_argument(
        "--raw-output-dir",
        default="raw_outputs",
        help="Directory to save per-image .npz raw output files",
    )
    parser.add_argument(
        "--quiet-per-image",
        action="store_true",
        help="Suppress the per-DMA 'Starting output DMA(s)/Waiting/Inference done' prints "
             "(keeps the progress bar clean on large batches)",
    )
    args = parser.parse_args()

    image_paths = list_images(args.data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")
    print(f"Found {len(image_paths)} image(s) in {args.data_dir}")

    finn = FINNXRT(
        args.xclbin,
        io_shape_dict=io_shape_dict,
        input_kernel_name=args.input_kernel,
        output_kernel_names=args.output_kernels,
        device_id=args.device_id,
    )

    img_size = (416, 416)

    n_total = len(image_paths)
    n_ok = 0
    failures = []  # list of (image_path, exception_str)
    exec_times = []  # per-image "exec_s" (device compute time, excludes host pre/post)
    total_times = []  # per-image total wall time including pre/post-processing

    iterator = enumerate(image_paths)
    if HAVE_TQDM:
        pbar = tqdm(total=n_total, unit="img", dynamic_ncols=True)
    else:
        pbar = None
        print(
            "tqdm not installed -- falling back to plain prints for progress. "
            "Install with `pip install tqdm` for a live progress bar."
        )

    run_start = time.time()

    for i, image_path in iterator:
        if pbar is None:
            print(f"\n[{i + 1}/{n_total}] Processing {image_path}")
        else:
            pbar.set_description(os.path.basename(image_path)[:30])

        try:
            img_t0 = time.time()
            driver_in, img_org, ratio, pad = preprocess_image(image_path, img_size)

            timing = {}
            outputs = finn.execute(
                driver_in, verbose=not args.quiet_per_image, timing=timing
            )

            # Sanity check: confirm shapes match expected oshape_normal
            for idx, out in enumerate(outputs):
                expected = io_shape_dict["oshape_normal"][idx]
                assert out.shape == expected, (
                    f"Output {idx} shape mismatch: got {out.shape}, expected {expected}"
                )

            img_stem = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(args.raw_output_dir, f"{img_stem}_raw.npz")

            save_raw_outputs(
                outputs,
                io_shape_dict,
                image_path,
                out_path,
                orig_shape=img_org.shape,
                ratio=ratio,
                pad=pad,
            )

            img_total = time.time() - img_t0
            n_ok += 1
            exec_times.append(timing["exec_s"])
            total_times.append(img_total)

            if pbar is not None:
                running_avg = sum(total_times) / len(total_times)
                fps = 1.0 / running_avg if running_avg > 0 else float("nan")
                pbar.set_postfix(
                    ok=n_ok,
                    fail=len(failures),
                    fps=f"{fps:.2f}",
                    dev_ms=f"{timing['exec_s'] * 1000:.1f}",
                )
            else:
                print(
                    f"  done in {img_total * 1000:.1f} ms "
                    f"(device: {timing['exec_s'] * 1000:.1f} ms) -> {out_path}"
                )

        except Exception as e:
            failures.append((image_path, str(e)))
            if pbar is not None:
                pbar.set_postfix(ok=n_ok, fail=len(failures))
                pbar.write(f"[FAILED] {image_path}: {e}")
            else:
                print(f"  [FAILED] {image_path}: {e}")
            continue

        finally:
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    run_total = time.time() - run_start

    print("\n" + "=" * 60)
    print("Run summary")
    print("=" * 60)
    print(f"Images found:     {n_total}")
    print(f"Succeeded:        {n_ok}")
    print(f"Failed:           {len(failures)}")
    print(f"Wall time:        {run_total:.2f} s")
    if total_times:
        avg_total = sum(total_times) / len(total_times)
        avg_exec = sum(exec_times) / len(exec_times)
        print(f"Avg time/image:   {avg_total * 1000:.1f} ms (host+device)")
        print(f"Avg device time:  {avg_exec * 1000:.1f} ms (accelerator only)")
        print(f"Throughput:       {1.0 / avg_total:.2f} img/s (end-to-end)")
    if failures:
        print("\nFailed images:")
        for path, err in failures:
            print(f"  - {path}: {err}")
    print(f"\nSaved {n_ok} raw output file(s) to {args.raw_output_dir}")

