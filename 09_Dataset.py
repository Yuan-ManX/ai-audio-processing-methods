# Dataset related content
# 0.Environment build torchaudio
import torchaudio
from torch.utils.data import Dataset
import os
import librosa
# from 07_STFT_tool import plot_waveform 

# 1.Download Dataset
audio_dataset = torchaudio.datasets.YESNO(root="./yes_no", download=True)

# 2.Define our dataset class
class AudioDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.audiofile = os.listdir(self.path)

    def __getitem__(self, index):
        audio_name = self.audiofile[index]
        wave_path = os.path.join(self.path, audio_name)
        waveform, sr = librosa.load(wave_path)
        return waveform, sr, audio_name

    def __len__(self):
        return len(self.audiofile)


# 3.Instantiate the audio dataset class
root_dir = r""
label_dir = ""
audio_dataset = AudioDataset(root_dir=root_dir, label_dir=label_dir)
index = 10
waveform, sr, name = audio_dataset[index]
# plot_waveform(waveform.T[:,0], sr)