# Summary Writer
# 0.Environment build
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from PIL import Image

# 1.load dataset tensor
image_path = r""
image = Image.open(image_path)

trans_tensor = torchvision.transforms.ToTensor()
image_tensor = trans_tensor(image)

# 2.image SummaryWriter
image_file = SummaryWriter("Image_log")
image_file.add_image(tag='Image', img_tensor=image_tensor)

# 3.image Normalize
trans_norm = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.3,0.3,0.3])
image_norm = trans_norm(image_tensor)
image_file.add_image(tag='Norm', img_tensor=image_norm)

# 4.image Resize 
trans_resize = torchvision.transforms.Resize(size=([200,200]))
image_resize = trans_resize(image_tensor)
image_file.add_image(tag='Resize', img_tensor=image_resize)

# 5.image RandomCrop
trans_randomcrop = torchvision.transforms.RandomCrop(size=(30,20))
image_randomcrop = trans_randomcrop(image_tensor)
image_file.add_image(tag='RandomCrop', img_tensor=image_randomcrop)

image_file.close()

audio_file = SummaryWriter('Audio_Log')
for i in range(20):
    audio_file.add_scalar(tag='y=2x', scalar_value=2*i, global_step=i)
    # audio_file.add_audio(tag='f"{name}"', snd_tensor=audio, sample_rate=sr)
audio_file.close()