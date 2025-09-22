import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

class VoiceAnalyzer:
    def __init__(self, fileName) -> None:
        self.file = fileName
        self.sampFreq, self.data = wavfile.read(f"exampleFiles/{fileName}.wav")
        self.nyquistFreq = self.sampFreq * 0.5

        if len(self.data.shape) == 1:
            self.audio = self.data
        else:
            self.audio = self.data[:, 0]

        self.sampNums = len(self.audio)
        self.audioTime = self.sampNums / self.sampFreq

    def audioPlot(self):
        timeAxis = np.linspace(0, self.audioTime, self.sampNums) 
        
        plt.plot(timeAxis, self.audio)
        plt.xlabel("Time (s)")
        plt.ylabel("Relative amplitude")
        plt.title("Audio in time domain")
        plt.show()

    def fft(self, filter=False, plot=False):
        signal = self.audio

        if filter: 
            # filtered signal 
            low = 75 / self.nyquistFreq
            high = 265 / self.nyquistFreq
            b, a = butter(3, [low, high], btype='bandpass')
            signal = filtfilt(b, a, self.audio.copy()) 

        # FFT original
        audio_fft = np.fft.fft(signal)
        N = len(audio_fft)
        freqs = np.fft.fftfreq(N, d=1/self.sampFreq)
        positive_freqs = freqs[:N//2]
        positive_magnitudes = np.abs(audio_fft[:N//2]) 
        peak_index = np.argmax(positive_magnitudes)

        if plot:
            plt.plot(positive_freqs, positive_magnitudes)
            plt.plot(positive_freqs[peak_index],\
            positive_magnitudes[peak_index],'ro',\
            label=f'f = {positive_freqs[peak_index]:.2f} Hz')
            
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Relative amplitude")
            plt.title("Audio in frequency domain")
            plt.xlim(0,500)
            plt.grid()
            plt.legend()
            plt.show()



if __name__ == "__main__":
    exampleWoman = VoiceAnalyzer("Marlene")
    exampleWoman.audioPlot()
    exampleWoman.fft(filter=True, plot=True)