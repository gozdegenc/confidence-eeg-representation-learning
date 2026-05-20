import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_eeg: torch.Tensor,
                z_periph: torch.Tensor) -> torch.Tensor:
        z_eeg    = F.normalize(z_eeg, dim=-1)
        z_periph = F.normalize(z_periph, dim=-1)
        sim_matrix = torch.matmul(z_eeg, z_periph.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_eeg    = F.cross_entropy(sim_matrix,   labels)
        loss_periph = F.cross_entropy(sim_matrix.T, labels)
        return (loss_eeg + loss_periph) / 2.0


class FusionModule(nn.Module):
    def __init__(self,
                 embed_dim: int = 128,
                 n_classes: int = 3,
                 behav_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # embed_dim*2 + behav_dim = 256 + 32 = 288
        fusion_input_dim = embed_dim * 2 + behav_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, z_eeg: torch.Tensor,
                z_periph: torch.Tensor,
                z_behav: Optional[torch.Tensor] = None) -> torch.Tensor:

        q = z_eeg.unsqueeze(1)
        k = z_periph.unsqueeze(1)
        attended, _ = self.cross_attention(q, k, k)
        attended = attended.squeeze(1)

        # EEG + attention çıktısı
        fused = torch.cat([z_eeg, attended], dim=-1)  # (batch, 256)

        # Behavioral ekle
        if z_behav is not None:
            fused = torch.cat([fused, z_behav], dim=-1)  # (batch, 288)
        else:
            # Behavioral yoksa sıfır vektör ekle
            pad = torch.zeros(fused.size(0), 32, device=fused.device)
            fused = torch.cat([fused, pad], dim=-1)

        return self.classifier(fused)