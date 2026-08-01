#!/bin/bash
# FINN+ build script for YOLOv8-OBB

# Define input and output file paths
MODEL_PATH="./exported.onnx"
CLEANED_MODEL_PATH="./exported_cleanup.onnx"
BUILD_FILE="build.yaml"

# Step 1: Run the Python cleanup script
echo "Running cleanup on $MODEL_PATH..."
python3 model_cleanup.py --model-path "$MODEL_PATH" --exported-filename "$CLEANED_MODEL_PATH"

# Check if cleanup was successful
if [ $? -ne 0 ]; then
    echo "Error during cleanup. Exiting..."
    exit 1
fi

# Step 2: Run finn build
echo "Running FINN+ YOLOv8-OBB build on $CLEANED_MODEL_PATH..."
finn build --skip-dep-update "$BUILD_FILE" "$CLEANED_MODEL_PATH"

# Check if finn build was successful
if [ $? -ne 0 ]; then
    echo "Error during finn build. Exiting..."
    exit 1
fi

echo "YOLOv8-OBB FINN+ build completed successfully!"
