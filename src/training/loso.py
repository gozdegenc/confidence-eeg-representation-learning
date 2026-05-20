
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from model import ConfidenceEEGModel

# ── Config ────────────────────────────────────────────
DATA_DIR          = Path('../../data/processed_cosubio')
N_SUBJECTS        = 34
N_EEG_CHANNELS    = 10
N_PERIPH_CHANNELS = 6
N_SAMPLES         = 512
N_CLASSES         = 3
BATCH_SIZE        = 64
EPOCHS            = 40
LR                = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Stabilize InfoNCE ─────────────────────────────────
class StableInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        sim = torch.clamp(sim, min=-20, max=20)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.T, labels)) / 2.0
        return loss


# ── Dataset ───────────────────────────────────────────
class LOSODataset(Dataset):
    def __init__(self, eeg, periph, labels, behavioral,
                 subject_ids, test_subject, split):
        mask = (subject_ids != test_subject) if split == 'train' \
               else (subject_ids == test_subject)
        self.eeg        = torch.from_numpy(eeg[mask]).float()
        self.periph     = torch.from_numpy(periph[mask]).float()
        self.labels     = torch.from_numpy(labels[mask]).long()
        self.behavioral = torch.from_numpy(behavioral[mask]).float()

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'eeg':        self.eeg[idx],
            'peripheral': self.periph[idx],
            'label':      self.labels[idx],
            'behavioral': self.behavioral[idx],
        }


# ── Tek fold egitimi ──────────────────────────────────
def train_one_subject(test_subject, eeg, periph, labels,
                      behavioral, subject_ids):
    print(f"\n{'─'*50}")
    print(f"Test subject: {test_subject:02d}")

    train_ds = LOSODataset(eeg, periph, labels, behavioral,
                           subject_ids, test_subject, 'train')
    test_ds  = LOSODataset(eeg, periph, labels, behavioral,
                           subject_ids, test_subject, 'test')

    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")
    print(f"  Test dagilim: {np.bincount(test_ds.labels.numpy())}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Sinif agirligi
    counts  = np.bincount(train_ds.labels.numpy(), minlength=N_CLASSES)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * N_CLASSES
    w_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    ce_fn    = nn.CrossEntropyLoss(weight=w_tensor)
    cl_fn    = StableInfoNCE(temperature=0.1).to(DEVICE)

    model = ConfidenceEEGModel(
        n_eeg_channels=N_EEG_CHANNELS,
        n_periph_channels=N_PERIPH_CHANNELS,
        n_samples=N_SAMPLES,
        n_classes=N_CLASSES,
        behav_dim=32,
        behav_input_dim=9,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            eeg_b    = batch['eeg'].to(DEVICE)
            periph_b = batch['peripheral'].to(DEVICE)
            labels_b = batch['label'].to(DEVICE)
            behav_b  = batch['behavioral'].to(DEVICE)
            # NaN/Inf temizle
            eeg_b    = torch.nan_to_num(eeg_b,    nan=0.0, posinf=1.0, neginf=-1.0)
            periph_b = torch.nan_to_num(periph_b, nan=0.0, posinf=1.0, neginf=-1.0)
            behav_b  = torch.nan_to_num(behav_b,  nan=0.0, posinf=1.0, neginf=-1.0)

            optimizer.zero_grad()
            z_eeg    = model.eeg_encoder(eeg_b)
            z_periph = model.peripheral_encoder(periph_b)
            z_behav  = model.behavioral_encoder(behav_b)
            logits   = model.fusion(z_eeg, z_periph, z_behav)
            loss_c   = cl_fn(z_eeg, z_periph)
            loss_e   = ce_fn(logits, labels_b)
            loss     = 0.5 * loss_c + 0.5 * loss_e

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        if epoch % 10 == 0:
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch['eeg'].to(DEVICE),
                                batch['peripheral'].to(DEVICE),
                                batch['behavioral'].to(DEVICE))
                    preds.extend(out['logits'].argmax(-1).cpu().numpy())
                    trues.extend(batch['label'].numpy())
            f1 = f1_score(trues, preds, average='macro', zero_division=0)
            unique = sorted(set(int(p) for p in preds))
            print(f"  Epoch {epoch:02d} | loss={avg_loss:.4f} | "
                  f"F1={f1:.3f} | preds={unique}")
            if f1 > best_f1:
                best_f1 = f1

    # Final test
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch['eeg'].to(DEVICE),
                        batch['peripheral'].to(DEVICE),
                        batch['behavioral'].to(DEVICE))
            preds.extend(out['logits'].argmax(-1).cpu().numpy())
            trues.extend(batch['label'].numpy())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average='macro', zero_division=0)
    print(f"  Final: Acc={acc:.3f} | F1={f1:.3f}")
    return acc, f1


# ── Ana fonksiyon ─────────────────────────────────────
def run_loso():
    print(f"Cihaz: {DEVICE}")
    print("Veri yukleniyor...")

    eeg         = np.load(DATA_DIR / 'eeg.npy').astype(np.float32)
    periph      = np.load(DATA_DIR / 'peripheral.npy').astype(np.float32)
    labels      = np.load(DATA_DIR / 'labels.npy').astype(np.int64)
    behavioral  = np.load(DATA_DIR / 'behavioral.npy').astype(np.float32)
    subject_ids = np.load(DATA_DIR / 'subject_ids.npy')

    print(f"Toplam epoch: {len(eeg)}")

    results  = {}
    all_acc, all_f1 = [], []

    for subj in np.unique(subject_ids):
        acc, f1 = train_one_subject(
            subj, eeg, periph, labels, behavioral, subject_ids)
        results[f'subject_{int(subj):02d}'] = {
            'acc': float(acc), 'f1': float(f1)}
        all_acc.append(acc)
        all_f1.append(f1)

    mean_acc = float(np.mean(all_acc))
    std_acc  = float(np.std(all_acc))
    mean_f1  = float(np.mean(all_f1))
    std_f1   = float(np.std(all_f1))

    print(f"\n{'='*50}")
    print(f"LOSO Sonuclari ({len(all_acc)} fold)")
    print(f"{'-'*50}")
    print(f"Accuracy : {mean_acc:.3f} +/- {std_acc:.3f}")
    print(f"Macro F1 : {mean_f1:.3f} +/- {std_f1:.3f}")
    print(f"{'='*50}")

    results['summary'] = {
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_f1':  mean_f1,  'std_f1':  std_f1,
        'n_folds':  len(all_acc),
    }
    out_path = Path('../../results/loso_results.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Sonuclar: {out_path}")


if __name__ == "__main__":
    run_loso()
