# Extract the bandwidth of the audio signal.

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

# 2.Calculate the bandwidth
sw_audio = librosa.feature.spectral_bandwidth(y=waveform, n_fft=1024).T[:, 0]

# 3.Plot the bandwidth information of audio
fig, aix = plt.subplots(2,2)
aix[0,0] = plt.plot(np.arange(0, len(sw_audio)), sw_audio, linewidth=1)
aix[0,0] = plt.set_title('Music')
fig.suptitle('Bandwidth')
plt.show()