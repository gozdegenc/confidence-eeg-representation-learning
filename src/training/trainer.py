
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))
from model import ConfidenceEEGModel


def train_one_epoch(model, loader, optimizer, device):
    """Bir epoch boyunca tüm batch'leri ileri-geri geçişle işler; ortalama kayıp, acc ve F1 döndürür."""
    model.train()
    losses, preds_all, labels_all = [], [], []

    for batch in loader:
        eeg    = batch['eeg'].to(device)
        periph = batch['peripheral'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        behavioral = batch.get('behavioral')
        if behavioral is not None:
            behavioral = behavioral.float().to(device)
        out = model(eeg, periph, behavioral, labels)
        out['loss'].backward()

        # Gradient patlamasını önler — InfoNCE + CE kaybı birlikte büyük gradyanlar üretebilir
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(out['loss'].item())
        preds_all.extend(out['logits'].argmax(dim=-1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return np.mean(losses), acc, f1


@torch.no_grad()
def evaluate(model, loader, device):
    """Gradyan hesaplamadan doğrulama/test setini değerlendirir; kayıp, acc ve F1 döndürür."""
    model.eval()
    losses, preds_all, labels_all = [], [], []

    for batch in loader:
        eeg    = batch['eeg'].to(device)
        periph = batch['peripheral'].to(device)
        labels = batch['label'].to(device)

        behavioral = batch.get('behavioral')
        if behavioral is not None:
            behavioral = behavioral.float().to(device)
        out = model(eeg, periph, behavioral, labels)
        losses.append(out['loss'].item())
        preds_all.extend(out['logits'].argmax(dim=-1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return np.mean(losses), acc, f1


def train(config: dict):
    """
    Ana eğitim fonksiyonu.
    config: tüm hiperparametreleri içeren dict
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # W&B başlat
    wandb.init(project="confidence-eeg", config=config)

    # Dataset ve DataLoader
    sys.path.append(str(Path(__file__).parent))
    from dataset import EEGConfidenceDataset

    data_dir = config['data_dir']
    train_ds = EEGConfidenceDataset(data_dir, split='train')
    val_ds   = EEGConfidenceDataset(data_dir, split='val')
    test_ds  = EEGConfidenceDataset(data_dir, split='test')

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model = ConfidenceEEGModel(
        n_eeg_channels   = config['n_eeg_channels'],
        n_periph_channels = config['n_periph_channels'],
        n_samples         = config['n_samples'],
        embed_dim         = config['embed_dim'],
        n_classes         = config['n_classes'],
        alpha             = config['alpha'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']   # L2 düzenleme — aşırı öğrenmeyi azaltır
    )
    # Öğrenme hızını kosinüs eğrisiyle sıfıra doğru düşürür — son epoch'larda ince ayar
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    # Checkpoint klasörü
    ckpt_dir = Path(config['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, device)
        scheduler.step()

        # W&B'ye kaydet
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss, 'train/acc': train_acc,
            'train/f1':   train_f1,
            'val/loss':   val_loss,   'val/acc':   val_acc,
            'val/f1':     val_f1,
            'lr': scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch:03d}/{config['epochs']} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} f1: {val_f1:.3f}")

        # En iyi modeli kaydet
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(),
                       ckpt_dir / 'best_model.pt')
            print(f"  → En iyi model kaydedildi (val f1: {val_f1:.3f})")

    # Test değerlendirmesi
    model.load_state_dict(torch.load(ckpt_dir / 'best_model.pt',
                                  weights_only=True))
    _, test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"\nTest sonucu → acc: {test_acc:.3f} | f1: {test_f1:.3f}")
    wandb.log({'test/acc': test_acc, 'test/f1': test_f1})
    wandb.finish()

    return model


# ── Çalıştırma ───────────────────────────────────────────────────────
if __name__ == "__main__":

    config = {
    'data_dir':           '../../data/processed_cosubio',  # ← gerçek veri
    'checkpoint_dir':     '../../experiments/checkpoints_cosubio',
    'n_eeg_channels':      10,   # ← Delta,Theta,Alpha1... (10 band)
    'n_periph_channels':    6,   # ← acc_x,acc_y,acc_z,bvp,eda,temp
    'n_samples':          512,   # ← 4 sn * 128 Hz
    'embed_dim':          128,
    'n_classes':            3,   # ← neutral / positive / negative
    'alpha':              0.5,
    'batch_size':          64,
    'epochs':              50,   # ← daha uzun eğit
    'lr':                3e-4,
    'weight_decay':       1e-4,
}

    train(config)