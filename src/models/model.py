import torch
import torch.nn as nn
from typing import Optional
from eeg_encoder import EEGNet
from peripheral_encoder import PeripheralEncoder
from behavioral_encoder import BehavioralEncoder
from contrastive_fusion import FusionModule, InfoNCELoss


class ConfidenceEEGModel(nn.Module):
    """
    Confidence-State Representation Learning — Güncellenmiş Model
    Üç modalite: EEG + Periferik + Behavioral Performance
    """

    def __init__(self,
                 n_eeg_channels: int = 10,
                 n_periph_channels: int = 6,
                 n_samples: int = 512,
                 embed_dim: int = 128,
                 behav_dim: int = 32,
                 behav_input_dim: int = 9,
                 n_classes: int = 3,
                 temperature: float = 0.07,
                 alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

        self.eeg_encoder = EEGNet(
            n_channels=n_eeg_channels,
            n_samples=n_samples,
            embed_dim=embed_dim,
        )
        self.peripheral_encoder = PeripheralEncoder(
            n_channels=n_periph_channels,
            n_samples=n_samples,
            embed_dim=embed_dim,
        )
        self.behavioral_encoder = BehavioralEncoder(
            input_dim=behav_input_dim,
            embed_dim=behav_dim,
        )
        self.fusion = FusionModule(
            embed_dim=embed_dim,
            n_classes=n_classes,
            behav_dim=behav_dim,
        )
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, eeg: torch.Tensor,
                peripheral: torch.Tensor,
                behavioral: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        z_eeg    = self.eeg_encoder(eeg)
        z_periph = self.peripheral_encoder(peripheral)
        z_behav  = self.behavioral_encoder(behavioral) \
                   if behavioral is not None else None

        logits = self.fusion(z_eeg, z_periph, z_behav)

        if labels is not None:
            loss_c = self.contrastive_loss(z_eeg, z_periph)
            loss_e = self.ce_loss(logits, labels)
            total  = self.alpha * loss_c + (1 - self.alpha) * loss_e
            return {
                'loss': total,
                'loss_contrastive': loss_c.item(),
                'loss_ce': loss_e.item(),
                'logits': logits,
            }
        return {
            'logits': logits,
            'z_eeg': z_eeg,
            'z_periph': z_periph,
            'z_behav': z_behav,
        }


# ── Test ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    model = ConfidenceEEGModel().to(device)

    eeg        = torch.randn(16, 10, 512).to(device)
    peripheral = torch.randn(16, 6, 512).to(device)
    behavioral = torch.randn(16, 9).to(device)
    labels     = torch.randint(0, 3, (16,)).to(device)

    out = model(eeg, peripheral, behavioral, labels)
    print(f"Loss:       {out['loss'].item():.4f}")
    print(f"Logits:     {out['logits'].shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parametreler: {total:,}")