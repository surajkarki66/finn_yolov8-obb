import numpy as np

from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper("./dataflow_parent.onnx")

mul_nodes = [n for n in model.graph.node if n.op_type == "Mul"]
add_nodes = [n for n in model.graph.node if n.op_type == "Add"]

for i, node in enumerate(mul_nodes):
    print(f"Mul node {i}: {node.name}")
    print("Inputs:", node.input)

for i, node in enumerate(add_nodes):
    print(f"Add node {i}: {node.name}")
    print("Inputs:", node.input)

# Extract Mul constants
mul0 = model.get_initializer("Mul_0_param0")
mul1 = model.get_initializer("Mul_1_param0")
mul2 = model.get_initializer("Mul_2_param0")

# Extract Add constants
add0 = model.get_initializer("Add_0_param0")
add1 = model.get_initializer("Add_1_param0")
add2 = model.get_initializer("Add_2_param0")

# Save as .npy
np.save("mul_0.npy", mul0)
np.save("mul_1.npy", mul1)
np.save("mul_2.npy", mul2)

np.save("add_0.npy", add0)
np.save("add_1.npy", add1)
np.save("add_2.npy", add2)

print("Mul_0 shape:", mul0.shape)
print("Mul_1 shape:", mul1.shape)
print("Mul_2 shape:", mul2.shape)

print("Add_0 shape:", add0.shape)
print("Add_1 shape:", add1.shape)
print("Add_2 shape:", add2.shape)