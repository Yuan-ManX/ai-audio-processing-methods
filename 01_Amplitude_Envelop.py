# Extract the amplitude envelope of the audio signal.

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

# 2.Calculate the amplitude envelope
def cal_amp_env(waveform, frame_length, hop_length):
    if len(waveform) % hop_length != 0:
        frame_num = int((len(waveform) - frame_length) / (frame_length - hop_length)) + 1
        pad_num = frame_num * hop_length + frame_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_num), mode="warp")
    frame_num = int((len(waveform) - frame_length) / (frame_length - hop_length)) + 1
    waveform_amp_env = []
    for i in range(frame_num):
        current_frame = waveform[i * (frame_length - hop_length) : i * (frame_length - hop_length) + frame_length]
        current_amp_env = max(current_frame)
        waveform_amp_env.append(current_amp_env)
    return np.array(waveform_amp_env)

# 3.Setting Parameters
frame_size = 1024
hop_size = int(frame_size * 0.5) # 512
audio_data = cal_amp_env(waveform=waveform, frame_length=frame_size, hop_length=hop_size)

# 4.Plot the amplitude envelope information of audio
frame_scale = np.arange(0, len(audio_data))
time_scale = librosa.frames_to_time(frame_scale, hop_length=hop_size)
plt.figure(figsize=(20,10))
librosa.display.waveshow(waveform)
plt.plot(time_scale, audio_data, color="y")
plt.title("Amplitude Envelope")
plt.show()