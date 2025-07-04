import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

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


if __name__ == "__main__":
    exampleWoman = VoiceAnalyzer("Marlene")
    exampleWoman.audioPlot()