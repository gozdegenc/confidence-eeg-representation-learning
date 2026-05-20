"""
EEGNet tabanlı EEG encoder — EEG sinyalinden 128 boyutlu gömme (embedding) üretir.
Kaynak: Lawhern et al. 2018, Journal of Neural Engineering.
3 aşamalı CNN: Temporal → Depthwise → Separable. Hafif mimari, 32+ kanallı EEG için optimize.
Modelin EEG kolunu oluşturur; PeripheralEncoder ile paralel çalışır.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet: EEG sinyalleri için hafif ve etkili CNN mimarisi.
    Kaynak: Lawhern et al., 2018

    Giriş:  (batch, n_channels, n_samples)  →  örn. (32, 32, 256)
    Çıkış:  (batch, embed_dim)              →  örn. (32, 128)
    """

    def __init__(self,
                 n_channels: int = 32,   # number of EEG channels
                 n_samples: int = 256,   # number of time points
                 embed_dim: int = 128,   # output embedding dimension
                 dropout: float = 0.5):
        super().__init__()

        # ── Block 1: Temporal convolution ─────────────────────────────
        # Captures temporal patterns in the signal
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),   # along time axis only
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(8),
        )

        # ── Block 2: Depthwise convolution (per-channel) ──────────────
        # Each EEG channel learns its own spatial pattern
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(n_channels, 1),  # collapses channel dimension
                groups=8,                     # depthwise
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # ── Block 3: Separable convolution ───────────────────────────
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                padding=(0, 8),
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # ── Projection head ───────────────────────────────────────────
        # Compute the flattened size after convolution blocks
        flatten_size = self._get_flatten_size(n_channels, n_samples)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _get_flatten_size(self, n_channels: int, n_samples: int) -> int:
        """Returns the number of neurons after the convolution blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.temporal_conv(dummy)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        x = x.unsqueeze(1)          # → (batch, 1, n_channels, n_samples)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.projection(x)      # → (batch, embed_dim)
        return x