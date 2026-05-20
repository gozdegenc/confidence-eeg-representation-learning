import torch
import torch.nn as nn


class BehavioralEncoder(nn.Module):
    """
    Game performance + self-esteem skorlarından embedding üretir.
    Girdi: (batch, 9)  →  Çıktı: (batch, embed_dim)
    """
    def __init__(self, input_dim: int = 9, embed_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)