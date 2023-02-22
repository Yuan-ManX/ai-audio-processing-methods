# Gradient content
# 0.Environment build
import torch
from torch import nn

# 1.Class
class Liner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=3, out_features=1), nn.Flatten(0,1))

    def forward(self, data):
        output = self.model(data)
        return output

# 2.Instantiate
x = torch.linspace(start=-torch.pi, end=torch.pi, steps=2000)
y = torch.sin(x)

p = torch.tensor([1,2,3])
input = x.unsqueeze(-1).pow(p)

net = Liner()
learning_rate = 1e - 6
for epoch in range(100):
    y_pred = net(input)
    loss = torch.square((y_pred-y),).sum()
    net.model.zero_grad()
    loss.backward()
    layer_liner = net.model[0]
    layer_flatten = net.model[1]

    with torch.no_grad():
        for parameter in net.model.parameters():
            parameter -= learning_rate * parameter.grad
    debug = 1
    if epoch % 20 == 0:
        print(loss)