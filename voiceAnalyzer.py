import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import plot

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
        timeArray = np.linspace(0, self.audioTime, self.sampNums) 