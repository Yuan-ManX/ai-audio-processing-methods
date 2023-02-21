# Nonliner content
# 0.Environment build
import torchvision.transforms
from PIL import Image
from torch import nn

# 1.image
image_path = r''
image = Image.open(image_path)
transform_tensor = torchvision.transforms.ToTensor()
image_tensor = transform_tensor(image)
print(image_tensor.shape)

# 2.Nonliner
class Nonliner(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonliner = nn.ReLU()

    def forward(self, data):
        output = self.nonliner(data)
        return output

# 3.instantiate
nonliner = Nonliner()
input = image_tensor.unsqueeze(0)
output = nonliner(input)
print(output.shape)

transform_PIL = torchvision.transforms.ToPILImage()
image_01 = transform_PIL(output.squeeze(0))
image_01.show()