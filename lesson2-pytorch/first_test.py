import torch
import numpy as np

t1 = torch.tensor([1, 2, 3])

print(t1)

t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(t2)


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
t1 = torch.rand_like(tensor, dtype=torch.float16)
print(t1.dtype)

# Convert tensor to numpy array
np_tensor = tensor.numpy()
print(np_tensor)

