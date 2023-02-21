# Liner content
# 0.Environment build
import torchvision
from torch import nn
from PIL import Image

# 1.image
image_path = r''
image = Image.open(image_path)

transform_tensor = torchvision.transforms.ToTensor()
image_tensor = transform_tensor(image)
print(image_tensor.shape)

# 2.Liner
class Liner(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner = nn.Linear(in_features=_, out_features=10)

    def forward(self, data):
        output = self.liner(data)

# 3.instantiate
liner = Liner()
input = image_tensor.unsqueeze(0)
output = liner(input)

transform_PIL = torchvision.transforms.ToPILImage()
image_01 = transform_PIL(output.sequeeze(0))
image_01.show()