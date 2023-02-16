# UrbanSound8K Dataset content
# 0.Environment build torchaudio
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os
import librosa
import torch

# 1.Define our dataset class
class UrbanSound8K(Dataset):
    def __init__(self, root_dir, csv_dir, transforms, target_samples, target_sr):
        super().__init__()
        self.root_dir = root_dir
        self.csv_dir = pd.read_csv(csv_dir)
        self.transforms = transforms
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
        file_fold = self.csv_dir.iloc[index, 5]
        audio_name = self.csv_dir[index, 0]
        target = self.csv_dir[index, 5]
        audio_path = os.path.join(self.root_dir, file_fold)
        self.audio_file = os.listdir(audio_path)
        wave_path = os.path.join(audio_path, audio_name)
        waveform, sr = librosa.load(wave_path)

        waveform = self.reshape_if_necessary(waveform)
        waveform = self.reshape_if_necessary(waveform, sr)
        waveform = self.mix_if_necessary(waveform)

        waveform_mel = self.transforms(waveform)

        return waveform_mel, sr, target
        # return waveform, sr, target

    def __len__(self):
        return len(self.audio_file)


# 2.Instantiate the audio dataset class
root_dir = r""
csv_dir = ""
trans_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024)
sound_dataset = UrbanSound8K(root_dir=root_dir, label_dir=csv_dir, transforms=trans_mel, target_samples=50000, target_sr=16000)
index = 10
waveform_mei, sr, target = sound_dataset[index]
# plot_waveform(waveform.T[:,0], sr)
# plot_spectrogram(waveform_mei[0,:,:])