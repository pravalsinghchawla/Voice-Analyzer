from scipy.io import wavfile

class VoiceAnalyzer:
    def __init__(self, fileName) -> None:

        self.file = fileName
        self.sampFreq, self.data = wavfile.read(f"Example_files/{fileName}.wav")
        self.nyquistFreq = self.sampFreq * 0.5

        if len(self.data.shape) == 1:
            self.audio = self.data
        else:
            self.audio = self.data[:, 0]

        self.sampNums = len(self.audio)
        self.audioTime = self.sampNums / self.sampFreq