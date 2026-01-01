from scipy.io import wavfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class WavData: 
    sr: int
    audio: np.ndarray
    duration_s: float


def load_wav(path: str | Path) -> WavData:
    path = Path(path)
    sr, data = wavfile.read(path)

    # Er dette nødvendig?
    if data.ndim == 1:
        audio = data
    else:
        audio = data[:, 0]

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    
    duration_s = float(len(audio) / sr)
    return WavData(sr=sr, audio=audio, duration_s=duration_s)

def extract_features(path: str | Path) -> np.ndarray:
    wav = load_wav(path)

    # 1) Enkel global energi-feature (støtte)
    rms = float(np.sqrt(np.mean(wav.audio ** 2)))

    # 2) F0-features fra FFT
    f0_feats, _ = extract_f0_features(
        wav.audio,
        wav.sr,
        frame_ms=40.0,
        hop_ms=10.0,
        fmin=70.0,
        fmax=300.0,
        rms_thresh=0.01
    )

    f0_mean = f0_feats["f0_mean"]
    f0_median = f0_feats["f0_median"]
    f0_std = f0_feats["f0_std"]

    if np.isnan(f0_mean):   f0_mean = 0.0
    if np.isnan(f0_median): f0_median = 0.0
    if np.isnan(f0_std):    f0_std = 0.0

    return np.array([rms, f0_mean, f0_median, f0_std], dtype=np.float32)


def frame_signal(audio: np.ndarray, sr: int, frame_ms: float = 40.0, hop_ms: float = 10):
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    
    if frame_len < 16:
        raise ValueError("frame_len is too small. Increase frame_ms")

    n_frames = 1 + max(0, (len(audio) - frame_len) // hop_len)
    for i in range(n_frames):
        start = i * hop_len
        yield audio[start:start + frame_len]

def estimate_f0_fft(frame: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 300.0) -> float:
    """
    Estimerer F0 fra én ramme ved å ta høyeste spektraltopp i [fmin, fmax].
    Returnerer np.nan hvis ingen fornuftig topp.
    """
    x = frame.astype(np.float32)
    x = x - np.mean(x)  

    w = np.hanning(len(x)).astype(np.float32)
    xw = x * w

    # FFT
    X = np.fft.rfft(xw)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(xw), d=1.0 / sr)

    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(idx) == 0:
        return np.nan

    k_peak = idx[np.argmax(mag[idx])]
    f_peak = freqs[k_peak]

    f_half = f_peak / 2.0
    if f_half >= fmin:
        k_half = np.argmin(np.abs(freqs - f_half))
        if mag[k_half] > 0.5 * mag[k_peak]:
            f_peak = freqs[k_half]

    return float(f_peak)

def extract_f0_features(audio: np.ndarray, sr: int,
                        frame_ms: float = 40.0, hop_ms: float = 10.0,
                        fmin: float = 70.0, fmax: float = 300.0,
                        rms_thresh: float = 0.01):

    f0_list = []
    for frame in frame_signal(audio, sr, frame_ms, hop_ms):
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        if rms < rms_thresh:
            f0_list.append(np.nan)
            continue

        f0 = estimate_f0_fft(frame, sr, fmin, fmax)
        f0_list.append(f0)

    f0_arr = np.array(f0_list, dtype=np.float32)
    valid = f0_arr[~np.isnan(f0_arr)]

    if len(valid) == 0:
        feats = {"f0_mean": np.nan, "f0_median": np.nan, "f0_std": np.nan}
        return feats, f0_arr

    feats = {
        "f0_mean": float(np.mean(valid)),
        "f0_median": float(np.median(valid)),
        "f0_std": float(np.std(valid, ddof=0)),
    }
    return feats, f0_arr
