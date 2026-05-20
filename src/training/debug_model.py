import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from model import ConfidenceEEGModel

device = torch.device('cuda')
model = ConfidenceEEGModel(
    n_eeg_channels=10, n_periph_channels=6,
    n_samples=512, behav_dim=32, behav_input_dim=9
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
ce = nn.CrossEntropyLoss()

for step in range(10):
    eeg    = torch.randn(64, 10, 512).to(device)
    periph = torch.randn(64,  6, 512).to(device)
    behav  = torch.randn(64,  9).to(device)
    labels = torch.randint(0, 3, (64,)).to(device)

    optimizer.zero_grad()
    z_eeg    = model.eeg_encoder(eeg)
    z_periph = model.peripheral_encoder(periph)
    z_behav  = model.behavioral_encoder(behav)
    logits   = model.fusion(z_eeg, z_periph, z_behav)
    loss_c   = model.contrastive_loss(z_eeg, z_periph)
    loss_e   = ce(logits, labels)
    loss     = 0.5 * loss_c + 0.5 * loss_e
    loss.backward()
    optimizer.step()

    preds = logits.argmax(-1)
    unique_preds = preds.unique().cpu().tolist()
    print(f"Step {step+1:02d} | loss={loss.item():.4f} | "
          f"tahmin edilen siniflar={unique_preds}")