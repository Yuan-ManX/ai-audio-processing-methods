# Pooling content
# 0.Environment build
import torch
import torchvision.transforms
from torch import nn
from PIL import Image

# 1.Image
image_path = r''
image = Image.open(image_path)
transform_tensor = torchvision.transforms.ToTensor()
image_tensor = transform_tensor(image)
print(image_tensor.shape)

# 2.Pooling 
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

    def forward(self, data):
        output = self.pooling(data)
        return output

# 3.instantiate
pooling = Pooling()
input = image_tensor.unsqueeze(0)
output = pooling(input)
print(output.shape)

transform_PIL = torchvision.transforms.ToPILImage()
image_01 = transform_PIL(output.squeeze(0))
image_01.show()