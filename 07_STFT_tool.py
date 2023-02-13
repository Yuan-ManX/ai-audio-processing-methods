# Extract the STFT-librosa of the audio signal.

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

# 2.Setting Parameters
n_fft = 1024
hop_size = 512
waveform_stft = librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_size)

# 3.Plot the stft information of audio
def plot_waveform(waveform, sr, title='Waveform'):
    waveform = np.arange(waveform)
    samples = waveform.size
    time_scale = np.linspace(0, samples/sr, num=samples)
    plt.figure(figsize=(20,10))
    plt.plot(time_scale, waveform, linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_waveform_stft(waveform, sr, n_fft, title='Waveform_FFT'):
    waveform = np.arange(waveform)
    samples = waveform.size
    waveform_fft = np.fft.rfft(waveform, n_fft)
    freq_scale = np.linspace(0, sr/2, num=int(n_fft/2)+1)
    plt.figure(figsize=(20,10))
    plt.plot(freq_scale, waveform_fft, linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()  

def plot_spectrogram(spectrogram, title='Spectrogram(db)'):
    plt.imshow(librosa.amplitude_to_db(spectrogram))
    plt.title(title)
    plt.xlabel('Frames/s')
    plt.ylabel('Frequency/Hz')
    plt.colorbar()
    plt.show()  



if __name__ == '__main__':

    plot_waveform(waveform=waveform, sr=sample_rate)
    plot_waveform_stft(waveform=waveform, sr=sample_rate, n_fft=1024)
    plot_spectrogram(np.abs(waveform_stft))