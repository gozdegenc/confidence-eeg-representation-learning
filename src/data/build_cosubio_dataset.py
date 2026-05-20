"""
CoSuBio preprocessing pipeline — subject_ids doğru üretiliyor.
Calistir: python build_cosubio_dataset.py
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from cosubio_loader import load_dataset_with_subject_ids
from preprocessor import preprocess_eeg, preprocess_peripheral

RAW_DIR       = "../../data/raw/cosubio"
PROCESSED_DIR = Path("../../data/processed_cosubio")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build():
    print("CoSuBio dataset isleniyor...")
    dataset = load_dataset_with_subject_ids(RAW_DIR)

    print(f"\nCikti shape'leri:")
    print(f"  EEG:        {dataset['eeg'].shape}")
    print(f"  Periferik:  {dataset['peripheral'].shape}")
    print(f"  Labels:     {dataset['labels'].shape}")
    print(f"  Subject IDs:{dataset['subject_ids'].shape}")
    print(f"  Sinif dagilimi: {np.bincount(dataset['labels'])}")

    # Subject bazinda epoch sayilarini goster
    for sid in np.unique(dataset['subject_ids']):
        n = np.sum(dataset['subject_ids'] == sid)
        print(f"  Subject {int(sid):02d}: {n} epoch")

    print("\nKaydediliyor...")
    np.save(PROCESSED_DIR / 'eeg.npy',         dataset['eeg'])
    np.save(PROCESSED_DIR / 'peripheral.npy',  dataset['peripheral'])
    np.save(PROCESSED_DIR / 'labels.npy',      dataset['labels'])
    np.save(PROCESSED_DIR / 'subject_ids.npy', dataset['subject_ids'])

    # Behavioral ekle
    from behavioral_loader import load_behavioral, assign_behavioral_to_epochs
    print("\nBehavioral veri yukleniyor...")
    behav_data = load_behavioral(RAW_DIR)
    behavioral_matrix = behav_data['behavioral']   # (34, 3, 9)

    behavioral_epochs = assign_behavioral_to_epochs(
        behavioral_matrix,
        dataset['subject_ids'],
        dataset['labels'])

    np.save(PROCESSED_DIR / 'behavioral.npy', behavioral_epochs)
    print(f"Behavioral: {behavioral_epochs.shape}")
    print(f"\nTamamlandi! -> {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    build()
