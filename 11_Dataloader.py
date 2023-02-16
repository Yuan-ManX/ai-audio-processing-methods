# DataLoader related content
from torch.utils.data import DataLoader, Dataset
import librosa
import os
import torch
import torchaudio

# 0.Define our dataset class
class AudioDateset(Dataset):
    def __init__(self, root_dir, label_dir, target_samples, target_sr):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.audio_file = os.listdir(self.path)
        self.target_samples = target_samples
        self.target_sr = target_sr

    def reshape_if_necessary(self, wavform):
        if waveform.shape[1] > self.target_samples:
            waveform = waveform[:, 0:self.target_samples]
        elif waveform.shape[1] < self.target_samples:
            num_pad = self.target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, pad=(0, num_pad))
        return waveform

    def reshape_if_necessary(self, waveform, sr):
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform=waveform, orig_freq=sr, new_freq=self.target_sr)
        return waveform

    def mix_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem__(self, index):
        audio_name = self.audio_file[index]
        audio_path = os.path.join(self.path, audio_name)
        waveform, sr = librosa.load(audio_path)

        waveform = self.reshape_if_necessary(waveform)
        waveform = self.reshape_if_necessary(waveform, sr)
        waveform = self.mix_if_necessary(waveform)

        return waveform, audio_name

    def __len__(self):
        return len(self.audio_file)


# 1.Set Batch Size, load dataset
root_dir = r""
label_dir = ""
audio_dataset = AudioDateset(root_dir=root_dir, label_dir=label_dir, target_samples=16000, target_sr=8000)
data_batch = DataLoader(dataset=audio_dataset, batch_size=100, shuffle=False, drop_last=False)

iteration = 0
for data in data_batch:
    waveform, audio_name = data
    iteration += 1
