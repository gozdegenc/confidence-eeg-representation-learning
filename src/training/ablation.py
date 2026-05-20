"""
Ablasyon deneyleri.

"""
import torch
import sys
import json
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent / 'models'))
sys.path.append(str(Path(__file__).parent))
from dataset import EEGConfidenceDataset
from model import ConfidenceEEGModel

# ── Sadece EEG kullanan basit model ──────────────────────────────────
# Ablasyon için: EEG encoder + doğrusal sınıflandırıcı. Periferik sinyal yok, contrastive yok.
import torch.nn as nn

class EEGOnlyModel(nn.Module):
    def __init__(self, n_channels=10, n_samples=512,
                 embed_dim=128, n_classes=3):
        super().__init__()
        from eeg_encoder import EEGNet
        self.encoder = EEGNet(n_channels, n_samples, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, eeg, peripheral=None, labels=None):  # noqa: ARG002  peripheral yok — DataLoader uyumluluğu
        z = self.encoder(eeg)
        logits = self.classifier(z)
        out = {'logits': logits}
        if labels is not None:
            out['loss'] = nn.CrossEntropyLoss()(logits, labels)
        return out


# ── Peripheral-only model ─────────────────────────────────────────────
# Ablasyon için: yalnızca EDA/BVP/sıcaklık/ivmeölçer kullanarak ne kadar sınıflandırılabilir?
class PeripheralOnlyModel(nn.Module):
    def __init__(self, n_channels=6, n_samples=512,
                 embed_dim=128, n_classes=3):
        super().__init__()
        from peripheral_encoder import PeripheralEncoder
        self.encoder = PeripheralEncoder(n_channels, n_samples, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, eeg=None, peripheral=None, labels=None):  # noqa: ARG002  eeg yok — DataLoader uyumluluğu
        z = self.encoder(peripheral)
        logits = self.classifier(z)
        out = {'logits': logits}
        if labels is not None:
            out['loss'] = nn.CrossEntropyLoss()(logits, labels)
        return out


# ── Contrastive loss olmayan fusion modeli ───────────────────────────
# Ablasyon için: EEG + periferik birleşimi var, InfoNCE kaybı yok.
# InfoNCE'nin katkısını ölçmek için tam modelle karşılaştırılır.
class FusionNoCLModel(nn.Module):
    def __init__(self, n_eeg=10, n_periph=6,
                 n_samples=512, embed_dim=128, n_classes=3):
        super().__init__()
        from eeg_encoder import EEGNet
        from peripheral_encoder import PeripheralEncoder
        from contrastive_fusion import FusionModule
        self.eeg_enc    = EEGNet(n_eeg, n_samples, embed_dim)
        self.periph_enc = PeripheralEncoder(n_periph, n_samples, embed_dim)
        self.fusion     = FusionModule(embed_dim, n_classes)
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, eeg, peripheral, labels=None):
        z_eeg    = self.eeg_enc(eeg)
        z_periph = self.periph_enc(peripheral)
        logits   = self.fusion(z_eeg, z_periph)
        out = {'logits': logits}
        if labels is not None:
            out['loss'] = self.ce(logits, labels)
        return out


# ── Eğitim & değerlendirme fonksiyonları ─────────────────────────────
def run_experiment(model, train_loader, val_loader,
                   test_loader, device, epochs=20, name=""):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # Eğitim
        model.train()
        for batch in train_loader:
            eeg    = batch['eeg'].to(device)
            periph = batch['peripheral'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            out = model(eeg, periph, labels)
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validasyon
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                eeg    = batch['eeg'].to(device)
                periph = batch['peripheral'].to(device)
                labels = batch['label'].to(device)
                out    = model(eeg, periph)
                preds_all.extend(
                    out['logits'].argmax(-1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        f1 = f1_score(labels_all, preds_all,
                      average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

        if epoch % 5 == 0:
            print(f"  [{name}] Epoch {epoch:02d}/{epochs} "
                  f"val f1: {f1:.3f}")

    # Test
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            eeg    = batch['eeg'].to(device)
            periph = batch['peripheral'].to(device)
            labels = batch['label'].to(device)
            out    = model(eeg, periph)
            preds_all.extend(out['logits'].argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    test_acc = accuracy_score(labels_all, preds_all)
    test_f1  = f1_score(labels_all, preds_all,
                         average='macro', zero_division=0)
    return test_acc, test_f1


# ── Ana fonksiyon ─────────────────────────────────────────────────────
def run_all_ablations():
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '../../data/processed_cosubio'
    epochs   = 20

    # DataLoader'ları hazırla
    def make_loaders():
        train_ds = EEGConfidenceDataset(data_dir, 'train')
        val_ds   = EEGConfidenceDataset(data_dir, 'val')
        test_ds  = EEGConfidenceDataset(data_dir, 'test')
        kw: dict[str, Any] = dict(batch_size=64, num_workers=0)
        return (DataLoader(train_ds, shuffle=True,  **kw),
                DataLoader(val_ds,   shuffle=False, **kw),
                DataLoader(test_ds,  shuffle=False, **kw))

    experiments = [
        ("EEG Only",       EEGOnlyModel(n_channels=10, n_samples=512)),
        ("Peripheral Only", PeripheralOnlyModel(n_channels=6, n_samples=512)),
        ("Fusion (no CL)", FusionNoCLModel(n_eeg=10, n_periph=6, n_samples=512)),
    ]

    # Ana modelimiz — CoSuBio checkpoint'inden yükle
    full_model = ConfidenceEEGModel(
        n_eeg_channels=10,
        n_periph_channels=6,
        n_samples=512,
    )
    ckpt = Path('../../experiments/checkpoints_cosubio/best_model.pt')
    full_model.load_state_dict(
        torch.load(ckpt, weights_only=True))
    full_model = full_model.to(device)

    results = {}

    # Baseline'ları çalıştır
    for name, model in experiments:
        print(f"\n{'─'*40}")
        print(f"Deney: {name}")
        model = model.to(device)
        loaders = make_loaders()
        acc, f1 = run_experiment(model, *loaders, device, epochs, name)
        results[name] = {'acc': acc, 'f1': f1}
        print(f"  → Test acc: {acc:.3f} | Test f1: {f1:.3f}")

    # Evaluate our main model (already trained)
    print(f"\n{'─'*40}")
    print("Experiment: Full Model (EEG + Peripheral + Contrastive)")
    _, _, test_loader = make_loaders()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            eeg    = batch['eeg'].to(device)
            periph = batch['peripheral'].to(device)
            labels = batch['label'].to(device)
            out    = full_model(eeg, periph)
            preds_all.extend(out['logits'].argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all,
                    average='macro', zero_division=0)
    results["Full Model (Ours)"] = {'acc': acc, 'f1': f1}
    print(f"  → Test acc: {acc:.3f} | Test f1: {f1:.3f}")

    # ── Sonuç tablosu ─────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'─'*50}")
    for name, res in results.items():
        marker = " ←" if name == "Full Model (Ours)" else ""
        print(f"{name:<30} {res['acc']:>10.3f} "
              f"{res['f1']:>10.3f}{marker}")
    print(f"{'═'*50}")

    # numpy kaydet — görselleştirmede kullanacağız
    out_path = Path('../../results/ablation_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSonuçlar kaydedildi: results/ablation_results.json")


if __name__ == "__main__":
    run_all_ablations()