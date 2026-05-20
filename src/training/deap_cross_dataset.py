
import numpy as np
from typing import cast, Tuple
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.signal import resample, butter, filtfilt
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from model import ConfidenceEEGModel

# ── Config ────────────────────────────────────────────
DEAP_DIR   = Path('../../data/raw/deap')
CKPT_PATH  = Path('../../experiments/checkpoints_cosubio/best_model.pt')
RESULT_DIR = Path('../../results')
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DEAP parametreleri
DEAP_EEG_FS     = 128   # Hz
DEAP_N_EEG      = 32    # kanal
DEAP_N_PERIPH   = 8     # kanal
DEAP_N_TRIALS   = 40    # video
DEAP_N_SAMPLES  = 8064  # 63 sn * 128 Hz

# Model beklentisi
TARGET_N_EEG    = 10    # CoSuBio EEG band sayisi gibi — PCA ile indirge
TARGET_N_PERIPH = 6     # EDA, BVP, TEMP, ACC-X, ACC-Y, ACC-Z
TARGET_SAMPLES  = 512   # 4 sn * 128 Hz
EPOCH_SAMPLES   = 512
EPOCH_STEP      = 256   # %50 overlap


# ── DEAP'i yukle ve donustur ──────────────────────────
def load_deap_subject(path: Path) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def bandpass(signal, low, high, fs=128, order=4):
    nyq = fs / 2
    b, a = cast(Tuple[np.ndarray, np.ndarray],
                butter(order, [low/nyq, high/nyq], btype='band'))
    return filtfilt(b, a, signal, axis=-1)


def extract_eeg_bands(eeg: np.ndarray, fs=128) -> np.ndarray:
    """
    32 kanal EEG'den 10 band gucu ozelligine donustur.
    CoSuBio'daki Delta,Theta,Alpha1,Alpha2,Beta1,Beta2,Gamma1,Gamma2,
    Attention(simulated),Meditation(simulated) yapisina uyum sagla.
    Cikti: (10, n_samples)
    """
    # Ortalama referans (tum kanallar ortalamasini cikar)
    eeg = eeg - eeg.mean(axis=0, keepdims=True)

    bands = {
        'Delta':  bandpass(eeg, 0.5, 4),
        'Theta':  bandpass(eeg, 4, 8),
        'Alpha1': bandpass(eeg, 8, 10),
        'Alpha2': bandpass(eeg, 10, 13),
        'Beta1':  bandpass(eeg, 13, 20),
        'Beta2':  bandpass(eeg, 20, 30),
        'Gamma1': bandpass(eeg, 30, 40),
        'Gamma2': bandpass(eeg, 40, 45),
    }

    # Her band icin tum kanallarin RMS gucunu hesapla → skaler sinyal
    band_powers = []
    for name, band_signal in bands.items():
        # (32, n_samples) -> (1, n_samples) — kanal ortalaması
        power = np.sqrt(np.mean(band_signal ** 2, axis=0, keepdims=True))
        band_powers.append(power)

    # Attention ve Meditation: Alpha/Beta oranlarindan simule et
    alpha = band_powers[2]  # Alpha1
    beta  = band_powers[4]  # Beta1
    attention  = beta  / (alpha + 1e-8)   # Beta/Alpha — dikkat
    meditation = alpha / (beta  + 1e-8)   # Alpha/Beta — meditasyon

    # Normalize
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8) * 100

    features = np.concatenate(band_powers + [norm(attention), norm(meditation)], axis=0)
    return features.astype(np.float32)  # (10, n_samples)


def select_periph_channels(periph: np.ndarray) -> np.ndarray:
    """
    DEAP periferik (8 kanal) → 6 kanal sec.
    DEAP kanallari: EDA(0), EMG(1), EOG(2,3), RESP(4), BVP(5), TEMP(6), ...
    CoSuBio kanallari: acc_x, acc_y, acc_z, bvp, eda, temperature
    Eslestir: EDA=0, BVP=5, TEMP=6, EMG yerine sifir dolgu (ACC yok)
    """
    eda  = periph[0:1]   # EDA
    bvp  = periph[5:6]   # BVP (plethysmograph)
    temp = periph[6:7]   # Temperature
    # ACC yok — sifir ile doldur (3 kanal)
    zeros = np.zeros((3, periph.shape[1]), dtype=np.float32)
    return np.concatenate([zeros, bvp, eda, temp], axis=0)  # (6, n_samples)


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def valence_to_label(valence_score: float) -> int:
    """
    DEAP valence (1-9) → 3 sinif
    <4: negative (confidence-reducing) = 2
    4-6: neutral = 0
    >6: positive (confidence-enhancing) = 1
    CoSuBio: 0=neutral, 1=positive, 2=negative
    """
    if valence_score < 4.0:
        return 2   # negative
    elif valence_score > 6.0:
        return 1   # positive
    else:
        return 0   # neutral


def process_deap() -> tuple:
    """
    Tum DEAP subject'lerini isle, epoch'la ve dondutr.
    """
    all_eeg, all_periph, all_labels = [], [], []

    for sid in range(1, 33):
        path = DEAP_DIR / f's{sid:02d}.dat'
        if not path.exists():
            print(f"  Atlandi: {path.name}")
            continue

        data   = load_deap_subject(path)
        eeg_raw    = data['data'][:, :32, :]   # (40, 32, 8064)
        periph_raw = data['data'][:, 32:, :]   # (40, 8, 8064)
        labels_raw = data['labels']             # (40, 4) — valence, arousal, dom, liking

        # Ilk 3 saniyeyi at (baseline), 60 sn veri al
        # DEAP'te ilk 3 sn baseline: 3*128=384 sample
        eeg_raw    = eeg_raw[:, :, 384:]
        periph_raw = periph_raw[:, :, 384:]

        for trial_idx in range(eeg_raw.shape[0]):
            eeg_trial    = eeg_raw[trial_idx]     # (32, n_samples)
            periph_trial = periph_raw[trial_idx]  # (8, n_samples)
            valence      = float(labels_raw[trial_idx, 0])
            label        = valence_to_label(valence)

            # EEG → 10 band feature
            eeg_feat = extract_eeg_bands(eeg_trial)   # (10, n_samples)
            eeg_feat = zscore(eeg_feat)

            # Periferik → 6 kanal
            per_feat = select_periph_channels(periph_trial.astype(np.float32))
            per_feat = zscore(per_feat)

            # Epoch'la
            n_samples = min(eeg_feat.shape[-1], per_feat.shape[-1])
            start = 0
            while start + EPOCH_SAMPLES <= n_samples:
                e = eeg_feat[:, start:start + EPOCH_SAMPLES]
                p = per_feat[:, start:start + EPOCH_SAMPLES]
                all_eeg.append(e)
                all_periph.append(p)
                all_labels.append(label)
                start += EPOCH_STEP

        print(f"  Subject {sid:02d}/32 islendi", end='\r')

    print(f"\nToplam DEAP epoch: {len(all_labels)}")
    print(f"Sinif dagilimi: {np.bincount(np.array(all_labels), minlength=3)}")

    return (np.stack(all_eeg).astype(np.float32),
            np.stack(all_periph).astype(np.float32),
            np.array(all_labels, dtype=np.int64))


# ── Dataset ───────────────────────────────────────────
class DEAPDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, periph, labels):
        self.eeg    = torch.from_numpy(eeg).float()
        self.periph = torch.from_numpy(periph).float()
        self.labels = torch.from_numpy(labels).long()
        # Behavioral yok — sifir vektoru kullan
        self.behav  = torch.zeros(len(labels), 9)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'eeg':        self.eeg[idx],
            'peripheral': self.periph[idx],
            'label':      self.labels[idx],
            'behavioral': self.behav[idx],
        }


# ── Ana test ──────────────────────────────────────────
def run_cross_dataset_test():
    print(f"Cihaz: {DEVICE}")
    print("="*50)
    print("DEAP Cross-Dataset Generalization Test")
    print("CoSuBio'da egitilen model → DEAP'te test")
    print("="*50)

    # Model yukle
    print(f"\nModel yukleniyor: {CKPT_PATH}")
    if not CKPT_PATH.exists():
        print(f"HATA: Checkpoint bulunamadi: {CKPT_PATH}")
        print("Mevcut checkpoint'leri kontrol et:")
        ckpt_dirs = list(Path('../../experiments').glob('*/best_model.pt'))
        for c in ckpt_dirs:
            print(f"  {c}")
        return

    model = ConfidenceEEGModel(
        n_eeg_channels=10,
        n_periph_channels=6,
        n_samples=512,
        behav_dim=32,
        behav_input_dim=9,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True,
                                      map_location=DEVICE))
    model.eval()
    print("Model yuklendi.")

    # DEAP isle
    print("\nDEAP verisi isleniyor...")
    eeg, periph, labels = process_deap()

    dataset = DEAPDataset(eeg, periph, labels)
    loader  = DataLoader(dataset, batch_size=128,
                         shuffle=False, num_workers=0)

    # Inference
    print("\nInference yapiliyor...")
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in loader:
            eeg_b   = torch.nan_to_num(batch['eeg'].to(DEVICE),
                                        nan=0.0, posinf=1.0, neginf=-1.0)
            per_b   = torch.nan_to_num(batch['peripheral'].to(DEVICE),
                                        nan=0.0, posinf=1.0, neginf=-1.0)
            beh_b   = batch['behavioral'].to(DEVICE)
            out     = model(eeg_b, per_b, beh_b)
            preds   = out['logits'].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(batch['label'].numpy())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    # Metrikler
    acc = accuracy_score(all_trues, all_preds)
    f1  = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    cm  = confusion_matrix(all_trues, all_preds)

    print(f"\n{'='*50}")
    print("SONUCLAR")
    print(f"{'-'*50}")
    print(f"Toplam epoch:    {len(all_preds)}")
    print(f"Test Accuracy:   {acc:.3f} ({acc*100:.1f}%)")
    print(f"Test Macro F1:   {f1:.3f}")
    print(f"Sinif dagilimi (gercek):  {np.bincount(all_trues, minlength=3)}")
    print(f"Sinif dagilimi (tahmin): {np.bincount(all_preds, minlength=3)}")
    print(f"\nKarisiklik Matrisi:")
    print(f"  (satir=gercek, sutun=tahmin)")
    print(f"  Siniflar: 0=Neutral, 1=Positive, 2=Negative")
    print(cm)
    print(f"{'='*50}")

    # Kaydet
    results = {
        'test_accuracy':  float(acc),
        'test_macro_f1':  float(f1),
        'n_epochs':       int(len(all_preds)),
        'class_dist_true': np.bincount(all_trues, minlength=3).tolist(),
        'class_dist_pred': np.bincount(all_preds, minlength=3).tolist(),
        'confusion_matrix': cm.tolist(),
        'source_dataset':   'CoSuBio',
        'target_dataset':   'DEAP',
        'protocol':         'zero-shot cross-dataset transfer',
    }

    out_path = RESULT_DIR / 'deap_cross_dataset_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSonuclar kaydedildi: {out_path}")

    return results


if __name__ == "__main__":
    run_cross_dataset_test()
