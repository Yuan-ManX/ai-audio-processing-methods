# Extract the STFT of the audio signal.

# 0.Setup Library
import librosa
import numpy as np
from matplotlib import pyplot as plt
import librosa.display

# 1.load audio
# absolute path
audio_path = r''
# relative path
audio_path_relative = ''
waveform, sample_rate = librosa.load(audio_path, sr=None)

# 2.zero padding
frame_size = 1024
hop_size = 512
if len(waveform) % hop_size != 0:
    frame_num = int((len(waveform) - frame_size) / hop_size) + 1
    pad_num = frame_num * hop_size + frame_size - len(waveform)
    waveform = np.pad(waveform, pad_width=(0, pad_num), mode="warp")
frame_num = int((len(waveform) - frame_size) / (frame_size - hop_size)) + 1

# Signal framing
row = np.tile(np.arange(0, frame_size), (frame_num, 1))
column = np.tile(np.arange(0, frame_num * (frame_size - hop_size), (frame_size - hop_size)), (frame_size, 1)).T
index = row + column
waveform_frame = waveform[index]

# window
waveform_frame = waveform_frame * np.hanning(frame_size)

# 3.STFT
n_fft = 1024
waveform_stft = np.fft.rfft(waveform_frame, n_fft)

# 4.power
waveform_pow = np.abs(waveform_stft)**2 / n_fft
waveform_db = 20 * np.log10(waveform_pow)

# 5.Plot the stft information of audio
plt.figure(figsize=(10,10))
plt.imshow(waveform_db)
y_ticks = np.arange(0, int(n_fft/2), 100)
plt.yticks(ticks=y_ticks, labels=y_ticks*sample_rate/n_fft)
plt.title("STFT")
plt.show()