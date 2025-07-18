import numpy as np

def FFT(audio: np.ndarray, sampFreq):

    fourierSignal = np.fft.fft(audio)
    nums = len(fourierSignal)
    freqs = np.fft.fftfreq(nums, d=1/sampFreq)
    positiveFreqs = freqs[:nums//2]
    positiveMags = np.abs(fourierSignal[:nums//2])
    dominantFreq_index = np.argmax(positiveMags)

    return positiveFreqs, positiveMags, dominantFreq_index