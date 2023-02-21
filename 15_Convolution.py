# Convolution Layer
# 0.Environment build
import torch.nn.functional
import torch
from torch import nn
from PIL import Image
import torchvision
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

# 1.conv1d
input = torch.tensor([[-1,2,1,1,0]])
kernel = torch.tensor([[1,0,-1]])

output = torch.nn.functional.conv1d(input=input, weight=kernel, stride=1, padding='Vaild')
print(output)
output_02 = torch.nn.functional.conv1d(input=input, weight=kernel, stride=1, padding=2)
print(output_02)


# 2.conv2d
input = torch.tensor([[[-1,2,1,1,0],
                        [1,2,3,4,5],
                        [5,6,7,8,9]]])
kernel = torch.tensor([[[1,0,-1],
                        [3,4,5],
                        [6,7,8]]])

output = torch.nn.functional.conv2d(input=input, weight=kernel, stride=1, padding='Vaild')
print(output.shape)
output_02 = torch.nn.functional.conv2d(input=input, weight=kernel, stride=2, padding=2)
print(output_02.shape)


# 3.image 
image_path = r''
image = Image(image_path)
transform_tensor = torchvision.transforms.ToTensor()
image_tensor = transform_tensor(image)
print(image_tensor.shape)
input = image_tensor.unsqueeze(0)

class Convolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, image_data):
        output = self.conv(image_data)
        return output


# 4.instantiate
image_file = SummaryWriter('Image_Convolution_log')
conv_nn = Convolution()
output = conv_nn(input)
print(output.shape)
transform_PIL = torchvision.transforms.ToPILImage()
image_01 = transform_PIL(output.squeeze(0))
image_01.show()
image_file.add_graph(conv_nn, input)
image_file.close()