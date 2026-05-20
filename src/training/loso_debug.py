"""
LOSO debug — model gercekten ogreniyor mu?

"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from model import ConfidenceEEGModel

DATA_DIR = Path('../../data/processed_cosubio')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veriyi yukle
eeg         = np.load(DATA_DIR / 'eeg.npy').astype(np.float32)
periph      = np.load(DATA_DIR / 'peripheral.npy').astype(np.float32)
labels      = np.load(DATA_DIR / 'labels.npy').astype(np.int64)
behavioral  = np.load(DATA_DIR / 'behavioral.npy').astype(np.float32)
subject_ids = np.load(DATA_DIR / 'subject_ids.npy')

# Subject 1 train/test ayir
test_mask  = subject_ids == 1
train_mask = subject_ids != 1

train_eeg   = torch.from_numpy(eeg[train_mask]).float()
train_per   = torch.from_numpy(periph[train_mask]).float()
train_beh   = torch.from_numpy(behavioral[train_mask]).float()
train_lab   = torch.from_numpy(labels[train_mask]).long()

test_eeg    = torch.from_numpy(eeg[test_mask]).float()
test_per    = torch.from_numpy(periph[test_mask]).float()
test_beh    = torch.from_numpy(behavioral[test_mask]).float()
test_lab    = torch.from_numpy(labels[test_mask]).long()

print(f"Train: {len(train_lab)} | Test: {len(test_lab)}")
print(f"Train dagilim: {np.bincount(train_lab.numpy())}")
print(f"Test  dagilim: {np.bincount(test_lab.numpy())}")
print(f"Behavioral unique degerler (ilk 3): {train_beh[:3, 0].numpy()}")
print()

# Model
model = ConfidenceEEGModel(
    n_eeg_channels=10, n_periph_channels=6,
    n_samples=512, behav_dim=32, behav_input_dim=9
).to(DEVICE)

# Sinif agirligi
counts  = np.bincount(train_lab.numpy(), minlength=3)
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum() * 3
w_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
ce_fn    = nn.CrossEntropyLoss(weight=w_tensor)
print(f"Sinif agirliklari: {weights.round(4)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# 5 epoch elle egit — gradient akiyor mu?
for epoch in range(1, 6):
    model.train()
    
    # Kucuk batch ile test
    idx = torch.randperm(len(train_lab))[:256]
    eeg_b   = train_eeg[idx].to(DEVICE)
    per_b   = train_per[idx].to(DEVICE)
    beh_b   = train_beh[idx].to(DEVICE)
    lab_b   = train_lab[idx].to(DEVICE)
    
    optimizer.zero_grad()
    z_eeg    = model.eeg_encoder(eeg_b)
    z_periph = model.peripheral_encoder(per_b)
    z_behav  = model.behavioral_encoder(beh_b)
    logits   = model.fusion(z_eeg, z_periph, z_behav)
    loss_c   = model.contrastive_loss(z_eeg, z_periph)
    loss_e   = ce_fn(logits, lab_b)
    loss     = 0.5 * loss_c + 0.5 * loss_e
    loss.backward()
    
    # Gradient norm kontrol
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    optimizer.step()
    
    preds = logits.argmax(-1)
    unique_preds = preds.unique().cpu().tolist()
    print(f"Epoch {epoch}: loss={loss.item():.4f} | "
          f"grad_norm={total_norm:.4f} | "
          f"unique_preds={unique_preds} | "
          f"CE={loss_e.item():.4f} | CL={loss_c.item():.4f}")

print()
print("Test sonucu (5 epoch sonra):")
model.eval()
preds, trues = [], []
with torch.no_grad():
    bs = 128
    for i in range(0, len(test_lab), bs):
        out = model(test_eeg[i:i+bs].to(DEVICE),
                    test_per[i:i+bs].to(DEVICE),
                    test_beh[i:i+bs].to(DEVICE))
        preds.extend(out['logits'].argmax(-1).cpu().numpy())
        trues.extend(test_lab[i:i+bs].numpy())

print(f"Unique predictions: {list(set(preds))}")
print(f"F1: {f1_score(trues, preds, average='macro', zero_division=0):.3f}")
print(f"Pred dagilim: {np.bincount(np.array(preds), minlength=3)}")
print(f"True dagilim: {np.bincount(np.array(trues), minlength=3)}")
