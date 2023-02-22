# Loss Content
# 0.Environment build
import numpy as np
import torch
from torch import nn

# 1.sine
x = np.linspace(start=np.pi, stop=np.pi, num=2000)
y = np.sin(x)

a,b,c,d = np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()
learning_rate = 1e - 6

# 2.test
for epoch in range(100):
    y_pred = a + b*x + c*x**2 + d*x**3
    loss = np.square(y_pred - y).sum()
    grad_y_pre = 2 * (y_pred - y)
    grad_a = grad_y_pre.sum()
    grad_b = (grad_y_pre * x**1).sum()
    grad_c = (grad_y_pre * x**2).sum()
    grad_d = (grad_y_pre * x**3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    if epoch % 20 == 0:
        print(loss)

# 3.loss function
y_pred = torch.tensor([1,2,3], dtype=float)
y = torch.tensor([1,2,10], dtype=float)

loss_l1 = torch.nn.L1Loss(reduction="sum")
result_01 = loss_l1(y_pred, y)
print(result_01)

loss_mse = torch.nn.MSELoss(reduction="sum")
result_02 = loss_mse(y_pred, y)
print(result_02)