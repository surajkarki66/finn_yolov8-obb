import argparse

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


def process_model(model_path, exported_filename):
    qonnx_model = ModelWrapper(model_path)

    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(InferDataTypes())
    qonnx_model = qonnx_model.transform(InferDataLayouts())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model = cleanup_model(qonnx_model)

    qonnx_model.save(exported_filename)
    print(f"Cleaned YOLOv8-OBB model saved to {exported_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up exported YOLOv8-OBB QONNX models for FINN+."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the input ONNX model file.",
    )
    parser.add_argument(
        "--exported-filename",
        type=str,
        required=True,
        help="Path where the cleaned ONNX model will be saved.",
    )
    args = parser.parse_args()
    process_model(args.model_path, args.exported_filename)


if __name__ == "__main__":
    main()
