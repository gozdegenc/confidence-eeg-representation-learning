"""
DEAP veri seti için tek seferlik ön işleme scripti.
Çalıştırma: python src/data/build_dataset.py
Ham .dat dosyalarını okur, filtreler, epoch'lar ve data/processed/ klasörüne .npy olarak kaydeder.
Bu script bir kez çalıştırılır; eğitim sırasında kaydedilen .npy dosyaları kullanılır.
"""
import numpy as np
from pathlib import Path
from deap_loader import load_all_subjects
from preprocessor import preprocess_eeg, preprocess_peripheral, binarize_labels

RAW_DIR       = "../../data/raw/deap"
PROCESSED_DIR = Path("../../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Veri yükleniyor...")
dataset = load_all_subjects(RAW_DIR, n_subjects=4)

# Shape: (32 subjects, 40 trials, 32 channels, 8064 samples)
eeg    = dataset['eeg']
periph = dataset['peripheral']
labels = dataset['labels']

print(f"Ham EEG shape: {eeg.shape}")
print(f"Raw peripheral shape: {periph.shape}")
print(f"Label shape: {labels.shape}")

# Her katılımcı ayrı ayrı işlenir; hafızada tümünü tutmak yerine birleştirilerek kaydedilir
all_eeg_proc, all_periph_proc, all_labels_proc = [], [], []

for subject_idx in range(eeg.shape[0]):
    print(f"Preprocessing: Subject {subject_idx+1:02d}/32", end='\r')

    s_eeg    = eeg[subject_idx]     # (40, 32, 8064) — 40 deneme
    s_periph = periph[subject_idx]  # (40, 8, 8064)
    s_labels = labels[subject_idx]  # (40, 4) — 4 boyutlu etiket

    eeg_proc    = preprocess_eeg(s_eeg)
    periph_proc = preprocess_peripheral(s_periph)

    # Epochlama her denemeyi birden fazla pencereye böler; etiketi her epoch için tekrarla
    n_epochs_per_trial = eeg_proc.shape[0] // s_labels.shape[0]
    labels_repeated = np.repeat(s_labels, n_epochs_per_trial, axis=0)
    
    all_eeg_proc.append(eeg_proc)
    all_periph_proc.append(periph_proc)
    all_labels_proc.append(labels_repeated)

print("\nKaydediliyor...")

np.save(PROCESSED_DIR / "eeg.npy",        np.concatenate(all_eeg_proc))
np.save(PROCESSED_DIR / "peripheral.npy", np.concatenate(all_periph_proc))
np.save(PROCESSED_DIR / "labels.npy",     np.concatenate(all_labels_proc))

print("Tamamlandı!")
print(f"EEG:        {np.load(PROCESSED_DIR / 'eeg.npy').shape}")
print(f"Peripheral: {np.load(PROCESSED_DIR / 'peripheral.npy').shape}")
print(f"Labels:     {np.load(PROCESSED_DIR / 'labels.npy').shape}")