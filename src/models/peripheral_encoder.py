
import torch
import torch.nn as nn


class PeripheralEncoder(nn.Module):
    """
    Periferik biyosignal encoder (EDA, BVP, sıcaklık, ivmeölçer).
    EEGNet ile paralel çalışır ve aynı boyutta embedding üretir.

    Giriş:  (batch, n_channels, n_samples)  →  örn. (32, 8, 256)
    Çıkış:  (batch, embed_dim)              →  örn. (32, 128)
    """

    def __init__(self,
                 n_channels: int = 8,
                 n_samples: int = 256,
                 embed_dim: int = 128,
                 dropout: float = 0.4):
        super().__init__()

        # ── Blok 1: Geniş kernel — düşük frekanslı biyosignal kalıplarını yakala ────
        self.block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),   # zaman boyutunu yarıya indir: 512 → 256
            nn.Dropout(dropout),
        )

        # ── Blok 2: Orta kernel — daha ince geçici örüntüler ────────────────────
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),   # 256 → 128
            nn.Dropout(dropout),
        )

        # ── Blok 3: Dar kernel + global havuzlama — boyutu embed_dim'e indir ─────
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),       # zaman boyutunu 1'e indir → (batch, 128, 1)
        )

        # ── Projeksiyon: düzleştir ve embed_dim boyutuna getir ──────────────────
        self.projection = nn.Sequential(
            nn.Flatten(),                  # (batch, 128, 1) → (batch, 128)
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),       # EEGNet çıkışıyla tutarlı ölçek
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.projection(x)   # → (batch, embed_dim)
        return x