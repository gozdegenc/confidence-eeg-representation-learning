"""
DEAP veri seti yükleyicisi.
DEAP: 32 katılımcı, 40 video deneyi, 40 EEG + 8 periferik kanal (EDA, EMG vb.)
Ham .dat dosyaları pickle formatında; latin1 encoding ile açılır.
"""
import pickle
import numpy as np
from pathlib import Path


def load_subject(data_dir: str, subject_id: int) -> dict:
    """
    Loads DEAP data for a single subject.
    subject_id: 1 to 32
    """
    path = Path(data_dir) / f"s{subject_id:02d}.dat"
    
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')  # DEAP .dat dosyaları latin1 ile kodlanmış

    # data: (40 trials, 40 channels, 8064 samples)
    # labels: (40 trials, 4) → valence, arousal, dominance, liking
    eeg_data   = raw['data'][:, :32, :]   # ilk 32 kanal: EEG elektrotları
    periph_data = raw['data'][:, 32:, :]  # son 8 kanal: periferik (EDA, EMG vb.)
    labels     = raw['labels']            # 4 etiket sütunu
    
    return {
        'eeg': eeg_data,           # (40, 32, 8064)
        'peripheral': periph_data, # (40, 8, 8064)
        'labels': labels           # (40, 4)
    }


def load_all_subjects(data_dir: str, n_subjects: int = 32) -> dict:
    """
    Loads and concatenates all subjects.
    """
    all_eeg, all_periph, all_labels = [], [], []
    
    for sid in range(1, n_subjects + 1):
        print(f"Loading: Subject {sid:02d}/{n_subjects}", end='\r')
        subject = load_subject(data_dir, sid)
        all_eeg.append(subject['eeg'])
        all_periph.append(subject['peripheral'])
        all_labels.append(subject['labels'])
    
    print("\nAll subjects loaded.")
    
    return {
        'eeg': np.stack(all_eeg),           # (32, 40, 32, 8064)
        'peripheral': np.stack(all_periph), # (32, 40, 8, 8064)
        'labels': np.stack(all_labels)      # (32, 40, 4)
    }