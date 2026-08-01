#!/usr/bin/env python3

import argparse

import numpy as np

from qonnx.core.modelwrapper import ModelWrapper

BRANCH_LABELS = [
    "angle_P3",
    "box_P3",
    "angle_P4",
    "box_P4",
    "angle_P5",
    "box_P5",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract Mul/Add dequant scale constants from YOLOv8-OBB dataflow_parent.onnx"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./dataflow_parent.onnx",
        help="Path to the dataflow_parent.onnx (6-output OBB graph)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to save mul_i.npy / add_i.npy",
    )
    args = parser.parse_args()

    model = ModelWrapper(args.model)

    mul_nodes = [n for n in model.graph.node if n.op_type == "Mul"]
    add_nodes = [n for n in model.graph.node if n.op_type == "Add"]

    print(f"Found {len(mul_nodes)} Mul nodes and {len(add_nodes)} Add nodes")
    for i, node in enumerate(mul_nodes):
        print(f"Mul node {i}: {node.name}, inputs: {list(node.input)}")
    for i, node in enumerate(add_nodes):
        print(f"Add node {i}: {node.name}, inputs: {list(node.input)}")

    if len(mul_nodes) != 6 or len(add_nodes) != 6:
        raise ValueError(
            f"Expected 6 Mul and 6 Add nodes for the OBB graph, got "
            f"{len(mul_nodes)} Mul and {len(add_nodes)} Add. Check the model."
        )

    print()
    for i in range(6):
        mul_name = f"Mul_{i}_param0"
        add_name = f"Add_{i}_param0"

        mul_val = model.get_initializer(mul_name)
        add_val = model.get_initializer(add_name)

        if mul_val is None:
            raise ValueError(f"Could not find initializer {mul_name}")
        if add_val is None:
            raise ValueError(f"Could not find initializer {add_name}")

        np.save(f"{args.out_dir}/mul_{i}.npy", mul_val)
        np.save(f"{args.out_dir}/add_{i}.npy", add_val)

        label = BRANCH_LABELS[i] if i < len(BRANCH_LABELS) else f"branch_{i}"
        print(f"[{i}] {label:10s}  mul shape: {mul_val.shape}  add shape: {add_val.shape}")

    print(f"\nSaved mul_0..5.npy and add_0..5.npy to {args.out_dir}")


if __name__ == "__main__":
    main()