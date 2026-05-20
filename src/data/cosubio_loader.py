"""
CoSuBio Dataset Loader — subject_ids ile birlikte
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample

# Sabitler
LABEL_MAP   = {'neutral': 0, 'positive': 1, 'negative': 2}
EEG_FS      = 1
PERIPH_FS   = 8
TARGET_FS   = 128
EPOCH_SEC   = 4
EEG_COLS    = ['Delta','Theta','Alpha1','Alpha2',
               'Beta1','Beta2','Gamma1','Gamma2',
               'Attention','Meditation']
PERIPH_COLS = ['acc_x','acc_y','acc_z','bvp','eda','temperature']


def load_dataset_with_subject_ids(data_dir: str | Path) -> dict:
    """
    CoSuBio'nun tüm verisini yukler, epoch'lar ve subject_ids döndürür.
    Her epoch hangi subject'e ait bilgisi korunur.
    """
    data_dir = Path(data_dir)

    print("EEG verisi okunuyor...")
    eeg_path = data_dir / 'eeg' / 'all_subjects_eeg.csv'
    eeg_df   = pd.read_csv(eeg_path)
    eeg_df['label'] = eeg_df['label'].map(LABEL_MAP)

    print("Reading peripheral signal (large file, please wait)...")
    periph_path = data_dir / 'Other biosignals' / 'merged_all_8Hz.csv'
    periph_df   = pd.read_csv(periph_path)
    periph_df['label'] = periph_df['label'].map(LABEL_MAP)

    print(f"EEG shape:     {eeg_df.shape}")
    print(f"Periph shape:  {periph_df.shape}")
    print(f"Subject'lar:   {sorted(eeg_df['sid'].unique())}")
    print(f"Label dagilimi (EEG):\n{eeg_df['label'].value_counts()}\n")

    all_eeg, all_periph, all_labels, all_subject_ids = [], [], [], []

    subjects = sorted(eeg_df['sid'].unique())

    for sid in subjects:
        print(f"  Subject {sid:02d}/{max(subjects)}", end='\r')

        eeg_sub    = eeg_df[eeg_df['sid'] == sid]
        periph_sub = periph_df[periph_df['sid'] == sid]

        for label_val in [0, 1, 2]:
            eeg_phase    = eeg_sub[eeg_sub['label'] == label_val]
            periph_phase = periph_sub[periph_sub['label'] == label_val]

            min_eeg_rows    = EPOCH_SEC * EEG_FS + 1
            min_periph_rows = EPOCH_SEC * PERIPH_FS + 1

            if len(eeg_phase) < min_eeg_rows:
                continue
            if len(periph_phase) < min_periph_rows:
                continue

            eeg_arr    = eeg_phase[EEG_COLS].values.T.astype(np.float32)
            periph_arr = periph_phase[PERIPH_COLS].values.T.astype(np.float32)

            if np.isnan(eeg_arr).any() or np.isnan(periph_arr).any():
                eeg_arr    = np.nan_to_num(eeg_arr,    nan=0.0)
                periph_arr = np.nan_to_num(periph_arr, nan=0.0)

            eeg_arr    = _zscore(eeg_arr)
            periph_arr = _zscore(periph_arr)
            eeg_arr    = np.nan_to_num(eeg_arr,    nan=0.0)
            periph_arr = np.nan_to_num(periph_arr, nan=0.0)

            eeg_rs    = _resample_to(eeg_arr,    EEG_FS,    TARGET_FS)
            periph_rs = _resample_to(periph_arr, PERIPH_FS, TARGET_FS)

            epoch_samples = int(EPOCH_SEC * TARGET_FS)
            step          = epoch_samples // 2

            min_len = min(eeg_rs.shape[-1], periph_rs.shape[-1])

            start = 0
            while start + epoch_samples <= min_len:
                e = eeg_rs[:,    start:start + epoch_samples]
                p = periph_rs[:, start:start + epoch_samples]

                all_eeg.append(e)
                all_periph.append(p)
                all_labels.append(label_val)
                all_subject_ids.append(int(sid))   # ← BURAYI EKLEDIK

                start += step

    print(f"\nTotal epochs: {len(all_labels)}")

    return {
        'eeg':         np.stack(all_eeg),
        'peripheral':  np.stack(all_periph),
        'labels':      np.array(all_labels,      dtype=np.int64),
        'subject_ids': np.array(all_subject_ids, dtype=np.int64),  # ← YENİ
    }


# Eski fonksiyon — geriye dönük uyumluluk
def load_dataset(data_dir: str | Path) -> dict:
    result = load_dataset_with_subject_ids(data_dir)
    return {
        'eeg':        result['eeg'],
        'peripheral': result['peripheral'],
        'labels':     result['labels'],
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1,  keepdims=True) + 1e-8
    return (x - mean) / std


def _resample_to(data: np.ndarray, orig_fs: int,
                 target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return data
    n_target = int(data.shape[-1] * target_fs / orig_fs)
    return np.asarray(resample(data, n_target, axis=-1), dtype=np.float32)
