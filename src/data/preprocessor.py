
from typing import cast

import numpy as np
from scipy.signal import butter, filtfilt


# ── Filtre fonksiyonları ──────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(order, [low, high], btype='band', output='ba'),
    )
    return filtfilt(b, a, data, axis=-1)


def extract_frequency_bands(eeg: np.ndarray, fs: float = 128.0) -> dict:
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta':  (13.0, 30.0),
        'gamma': (30.0, 45.0),
    }
    result = {}
    for band_name, (low, high) in bands.items():
        result[band_name] = bandpass_filter(eeg, low, high, fs)
    return result


# ── Normalizasyon ────────────────────────────────────────────────────

def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    if method == 'zscore':
        mean = data.mean(axis=-1, keepdims=True)
        std  = data.std(axis=-1, keepdims=True) + 1e-8  # prevent division by zero
        return (data - mean) / std
    elif method == 'minmax':
        min_val = data.min(axis=-1, keepdims=True)
        max_val = data.max(axis=-1, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Epoch'lama ───────────────────────────────────────────────────────

def create_epochs(data: np.ndarray, epoch_len_sec: float,
                  overlap: float, fs: float = 128.0) -> np.ndarray:
    epoch_samples = int(epoch_len_sec * fs)
    step = int(epoch_samples * (1 - overlap))
    n_samples = data.shape[-1]
    
    epochs = []
    start = 0
    while start + epoch_samples <= n_samples:
        epochs.append(data[..., start:start + epoch_samples])
        start += step
    
    return np.stack(epochs)  # (n_epochs, n_channels, epoch_samples)


# ── Etiket dönüşümü ──────────────────────────────────────────────────

def binarize_labels(labels: np.ndarray, threshold: float = 4.5) -> np.ndarray:
    return (labels >= threshold).astype(np.int64)


# ── Ana pipeline ─────────────────────────────────────────────────────

def preprocess_eeg(eeg: np.ndarray, fs: float = 128.0,
                   epoch_len: float = 2.0, overlap: float = 0.5) -> np.ndarray:
    all_epochs = []
    
    for trial_idx in range(eeg.shape[0]):
        trial = eeg[trial_idx]  # (n_channels, n_samples)
        
        # 1. Bandpass filter (keep 0.5–45 Hz, discard noise outside)
        filtered = bandpass_filter(trial, lowcut=0.5, highcut=45.0, fs=fs)

        # 2. Normalize
        normalized = normalize_signal(filtered, method='zscore')

        # 3. Epoch
        epochs = create_epochs(normalized, epoch_len_sec=epoch_len,
                               overlap=overlap, fs=fs)
        # epochs: (n_epochs, n_channels, epoch_samples)
        all_epochs.append(epochs)
    
    return np.concatenate(all_epochs, axis=0)


def preprocess_peripheral(periph: np.ndarray, fs: float = 128.0,
                           epoch_len: float = 2.0,
                           overlap: float = 0.5) -> np.ndarray:
    all_epochs = []
    
    for trial_idx in range(periph.shape[0]):
        trial = periph[trial_idx]
        
        # Lower bandpass for peripheral signals
        filtered = bandpass_filter(trial, lowcut=0.01, highcut=30.0, fs=fs)
        normalized = normalize_signal(filtered, method='zscore')
        epochs = create_epochs(normalized, epoch_len_sec=epoch_len,
                               overlap=overlap, fs=fs)
        all_epochs.append(epochs)
    
    return np.concatenate(all_epochs, axis=0)
