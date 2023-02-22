# Model development process
# 0.Environment build
from torch import nn
import torch

# 1.Dataset
input_data = torch.ones(size=(3,3,32,32))
y = torch.tensor([1,2,3,4,5,6,7,8,9])

# 2.Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3,32,5,1,2),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(32,64,5,1,2),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(64,128,5,1,2),
                                    nn.MaxPool2d(2),
                                    nn.Flatten(),
                                    nn.Linear(128*4*4, 128),
                                    nn.Linear(128, 10))
    def forward(self, data):
        output_data = self.model(data)
        return output_data

# 3.Instantiate
net = NeuralNetwork()
loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.RMSprop(params=net.model.parameters(), lr=1e-4)


for epoch in range(100):
    y_pred = net(input_data)

    # 4.loss
    loss = loss_fn(y_pred, y)

    # 5.update optimizer weight
    optimizer.zero_grad()

    # 6.gradient
    loss.backward()

    # 7.optimizer
    optimizer.step()

    if epoch % 10 == 0:
        print(loss)


