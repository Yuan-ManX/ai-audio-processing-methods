# Tensor related content
import torch
import numpy as np


# 0.Data type of tensor
tensor_int = torch.IntTensor([1,2,3])
tensor_float = torch.DoubleTensor([1,2,3])

# only defines the shape without assigning a value
tensor_size_01 = torch.Tensor((3,3,3))
tensor_size_02 = torch.Tensor(3,3,3)

# Define a tensor of all 0, similar in shape to input
tensor_zero = torch.zeros_like(tensor_size_02)

# Define elements as a set of random numbers distributed in [0,1]
tensor_rand = torch.rand((4,4))

# Define elements as a set of random numbers that satisfy a normal distribution (mean 0, variance 1)
tensor_random = torch.randn((4,4))

# Returns an integer whose shape is 1*n and whose elements are not repeated within 0-n
tensor_rand_int = torch.randperm(5)


# 1.Tensor reshaping
tensor_3d = tensor_size_02.view(1,4,8)
tensor_3d_02 = tensor_size_02.resize(1,8,4)


# 2.Data type of tensor, arbitrary position adjustment
tensor_float_int = tensor_float.int()
x = tensor_float_int[1]


# 3.Tensor operations
a = torch.tensor((1,2,3))
b = torch.tensor((4,5,6))
a_add_b = torch.add(a,b)
a_sub_b = torch.sub(b,a)
a_mul_b = torch.mul(a,b)
a_div_b = torch.div(b,a)
a_3 = a.pow(3)
a_matrix_mul_b = torch.matmul(a,b.T)
a_matrix_mul_b_02 = torch.matmul(a,b)


# 4.Stitching of tensors, horizontal and vertical
a_append_b = torch.cat((a,b), dim=0)


# 5.Convert list and array types to tensors
# np -> tensor
num = np.array((1,2,3,4,5))
num_tensor = torch.tensor(num)
num_tensor_01 = torch.from_numpy(num)

# tensor -> np
tensor_a = torch.tensor((1,2,3))
tensor_num = tensor_a.numpy()
