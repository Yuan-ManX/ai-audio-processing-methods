# Nerual Network
# 0.Environment build
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

# 1.NN
class AudioNerualNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=10,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
        self.pooling3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.line1 = nn.Linear(in_features=64*4*4, out_features=64)
        self.line2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, audio):
        output = self.conv1(audio)
        output = self.pooling1(output)
        output = self.conv2(output)
        output = self.pooling2(output)
        output = self.conv3(output)
        output = self.pooling3(output)
        flatten = self.flatten(output)
        liner = self.line1(flatten)
        liner = self.line2(liner)
        return liner

# 2.Instantiate the audio moulde class
audio_test = AudioNerualNetwork()
print(audio_test)
input_data = torch.Tensor(1,10,32,32)
output_data = audio_test(input_data)
print(output_data.shape)

# 3.Summary Writer
audio_file = SummaryWriter('Audio_Nerual_Network_Log')
audio_file.add_graph(model=audio_test, input_to_model=input_data)
audio_file.close()


# 4.Sequential Net
class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels=10,out_channels=32,kernel_size=5,stride=1,padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Flatten(),
                                    nn.Linear(in_features=64*4*4, out_features=64),
                                    nn.Linear(in_features=64, out_features=10))
    
    def forward(self, audio):
        output = self.model(audio)
        return output

# 5.Instantiate the audio net moulde class
audio_net = AudioNet()
print(audio_net)
input_data_02 = torch.Tensor(1,10,32,32)
output_data_02 = audio_net(input_data_02)
print(output_data_02.shape)

# 6.Summary Writer
audio_file_02 = SummaryWriter('Audio_Net_Log')
audio_file_02.add_graph(model=audio_net, input_to_model=input_data_02)
audio_file_02.close()