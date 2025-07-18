import matplotlib.pyplot as plt
import numpy as np


def timePlot(time: np.ndarray, audio: np.ndarray):
    
    plt.plot(time, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative amplitude")
    plt.title("Audio in time domain")
    plt.show()

def freqPlot(freqs: np.ndarray, mags: np.ndarray):

    plt.plot(freqs, mags)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative magnitude")
    plt.title("Audio in frequency domain")
    plt.show()
