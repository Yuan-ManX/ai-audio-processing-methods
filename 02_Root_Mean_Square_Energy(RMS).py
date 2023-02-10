# Extract the RMS of the audio signal.

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

# 2.Calculate the RMS
def cal_RMS(waveform, frame_length, hop_length):
    if len(waveform) % hop_length != 0:
        frame_num = int((len(waveform) - frame_length) /  hop_length) + 1
        pad_num = frame_num * hop_length + frame_length - len(waveform)
        waveform = np.pad(waveform, pad_width=(0, pad_num), mode="warp")
    frame_num = int((len(waveform) - frame_length) / hop_length) + 1
    waveform_rms = []
    for i in range(frame_num):
        current_frame = waveform[i * (frame_length - hop_length) : i * (frame_length - hop_length) + frame_length]
        current_rms = np.sqrt(np.sum(current_frame**2) / frame_length)
        waveform_rms.append(current_rms)
    return np.array(waveform_rms)

# 3.Setting Parameters
frame_size = 1024
hop_size = int(frame_size * 0.5) # 512
audio_data = cal_RMS(waveform=waveform, frame_length=frame_size, hop_length=hop_size)

# 4.Plot the RMS information of audio
frame_scale = np.arange(0, len(audio_data), step=1)
time_scale = librosa.frames_to_time(frame_scale, hop_length=hop_size)
plt.figure(figsize=(20,10))
librosa.display.waveshow(waveform)
plt.plot(time_scale, audio_data, color="r")
plt.title("RMS[Root Mean Sqare Energy]")
plt.show()

# 5.librosa.feature.rms 
audio_data_librosa = librosa.feature.rms(y=waveform, frame_length=frame_size, hop_length=hop_size).T[1:, 0]
plt.figure(figsize=(20,10))
librosa.display.waveshow(waveform)
plt.plot(time_scale, audio_data, color="r")
plt.title("RMS[Root Mean Sqare Energy]-Librosa")
plt.show()

bias = audio_data_librosa - audio_data
print(f'the bias is {bias} \n Success!')
