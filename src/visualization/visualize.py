"""
Sunum için tüm görselleştirmeleri üretir ve results/figures/ klasörüne kaydeder.

  fig1_ablation.png       — model bileşenlerinin katkı karşılaştırması
  fig2_tsne.png           — EEG ve periferik embedding uzayları (t-SNE ile 2B)
  fig3_confusion_matrix.png — test seti sınıflandırma hataları
  fig4_training_curves.png  — kayıp ve F1 eğitim eğrileri
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # ekransız çalıştırma
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))
sys.path.append(str(Path(__file__).parent.parent / 'training'))

from model import ConfidenceEEGModel
from dataset import EEGConfidenceDataset

# ── Genel stil ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.dpi':         150,
})

COLORS  = ['#5C6BC0', '#26A69A', '#EF5350']   # 3 sınıf rengi
LABELS  = ['Low Confidence', 'Neutral', 'High Confidence']
FIG_DIR = Path('../../results/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '../../data/processed_cosubio'
CKPT     = '../../experiments/checkpoints_cosubio/best_model.pt'

# ── Model ve embedding'leri yükle ────────────────────────────────────
def load_model_and_embeddings():
    model = ConfidenceEEGModel(
    n_eeg_channels=10,
    n_periph_channels=6,
    n_samples=512,
)
    model.load_state_dict(torch.load(CKPT, weights_only=True))
    model = model.to(DEVICE).eval()

    test_ds     = EEGConfidenceDataset(DATA_DIR, 'test')
    test_loader = DataLoader(test_ds, batch_size=128,
                             shuffle=False, num_workers=0)

    z_eegs, z_periphs, all_preds, all_labels = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            eeg    = batch['eeg'].to(DEVICE)
            periph = batch['peripheral'].to(DEVICE)
            labels = batch['label']

            out = model(eeg, periph)
            z_eegs.append(out['z_eeg'].cpu().numpy())
            z_periphs.append(out['z_periph'].cpu().numpy())
            all_preds.extend(out['logits'].argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return (np.concatenate(z_eegs),
            np.concatenate(z_periphs),
            np.array(all_preds),
            np.array(all_labels))


# ── Figür 1: Ablasyon karşılaştırma çubuğu ───────────────────────────
def plot_ablation():
    models    = ['EEG\nOnly', 'Peripheral\nOnly',
                 'Fusion\n(no CL)', 'Full Model\n(Ours)']
    accuracy  = [0.377, 0.591, 0.663, 0.676]
    f1_scores = [0.206, 0.574, 0.651, 0.670]

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w/2, accuracy,  w, label='Accuracy',
                   color='#5C6BC0', alpha=0.85)
    bars2 = ax.bar(x + w/2, f1_scores, w, label='Macro F1',
                   color='#26A69A', alpha=0.85)

    # Değerleri çubukların üstüne yaz
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9)

    # Full model çerçevesi
    ax.axvspan(2.5, 3.5, alpha=0.06, color='#EF5350')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.0, 0.65)
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Component Contribution Analysis',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper left')
    ax.axhline(1/3, color='gray', linestyle='--',
               linewidth=0.8, label='Random baseline (0.333)')

    plt.tight_layout()
    path = FIG_DIR / 'fig1_ablation.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {path}")


# ── Figür 2: t-SNE — EEG embedding uzayı ────────────────────────────
def plot_tsne(z_eeg, z_periph, labels):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (z, title) in zip(axes, [
        (z_eeg,    'EEG Embedding Space (t-SNE)'),
        (z_periph, 'Peripheral Embedding Space (t-SNE)'),
    ]):
        print(f"  t-SNE hesaplanıyor: {title}...")
        z_2d = TSNE(n_components=2, perplexity=30,
            random_state=42, max_iter=500).fit_transform(z)

        for cls_idx, (color, label) in enumerate(zip(COLORS, LABELS)):
            mask = labels == cls_idx
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=color, label=label, alpha=0.55,
                       s=18, linewidths=0)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.legend(fontsize=9, markerscale=1.5)

    plt.suptitle('Latent Space Visualization by Confidence State',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = FIG_DIR / 'fig2_tsne.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {path}")


# ── Figür 3: Confusion matrix ────────────────────────────────────────
def plot_confusion_matrix(preds, labels):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Low\nConf.', 'Neutral', 'High\nConf.'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')

    ax.set_title('Confusion Matrix — Test Set',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    path = FIG_DIR / 'fig3_confusion_matrix.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {path}")


# ── Figür 4: Eğitim eğrileri (W&B'den manuel) ────────────────────────
def plot_training_curves():
    # W&B'den çekilen değerleri manuel girildi
    epochs     = list(range(1, 51))
    train_loss = [2.6557,2.5858,2.5556,2.5421,2.5279,
                  2.5149,2.5052,2.4970,2.4898,2.4827,
                  2.4745,2.4670,2.4601,2.4564,2.4504,
                  2.4413,2.4367,2.4313,2.4265,2.4210,
                  2.4160,2.4128,2.4052,2.4012,2.3971,
                  2.3938,2.3899,2.3856,2.3834,2.3803,
                  2.3760,2.3720,2.3696,2.3664,2.3613,
                  2.3609,2.3613,2.3584,2.3548,2.3522,
                  2.3506,2.3489,2.3503,2.3459,2.3469,
                  2.3466,2.3446,2.3443,2.3444,2.3434]
    val_loss   = [2.6211,2.5578,2.5481,2.5255,2.5196,
                  2.5114,2.4905,2.4927,2.4862,2.4653,
                  2.4645,2.4532,2.4362,2.4425,2.4277,
                  2.4146,2.4189,2.4197,2.4042,2.3944,
                  2.3886,2.3919,2.3883,2.3747,2.3882,
                  2.3672,2.3638,2.3619,2.3528,2.3585,
                  2.3482,2.3429,2.3520,2.3484,2.3402,
                  2.3410,2.3473,2.3359,2.3391,2.3384,
                  2.3402,2.3349,2.3337,2.3319,2.3358,
                  2.3341,2.3379,2.3340,2.3371,2.3386]
    val_f1     = [0.409,0.487,0.494,0.537,0.536,
                  0.543,0.562,0.562,0.572,0.593,
                  0.594,0.612,0.618,0.620,0.635,
                  0.636,0.632,0.628,0.645,0.657,
                  0.655,0.653,0.659,0.669,0.648,
                  0.670,0.675,0.678,0.682,0.680,
                  0.687,0.699,0.682,0.684,0.694,
                  0.697,0.688,0.703,0.696,0.691,
                  0.692,0.699,0.696,0.700,0.694,
                  0.697,0.693,0.699,0.691,0.694]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss eğrisi
    ax = axes[0]
    ax.plot(epochs, train_loss, color='#5C6BC0',
            label='Train loss', linewidth=2)
    ax.plot(epochs, val_loss,   color='#EF5350',
            label='Val loss',   linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss',
                 fontsize=12, fontweight='bold')
    ax.legend()

    # F1 eğrisi
    ax = axes[1]
    ax.plot(epochs, val_f1, color='#26A69A',
            linewidth=2, label='Val Macro F1')
    best_epoch = int(np.argmax(val_f1)) + 1
    best_f1    = max(val_f1)
    ax.axvline(best_epoch, color='gray',
               linestyle=':', linewidth=1.2)
    ax.scatter([best_epoch], [best_f1],
               color='#EF5350', zorder=5, s=60,
               label=f'Best (epoch {best_epoch}, F1={best_f1:.3f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Macro F1')
    ax.set_title('Validation Macro F1 Score',
                 fontsize=12, fontweight='bold')
    ax.legend()

    plt.suptitle('Training Dynamics',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = FIG_DIR / 'fig4_training_curves.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {path}")


# ── Ana akış ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Figürler üretiliyor...\n")

    print("[1/4] Ablasyon grafiği...")
    plot_ablation()

    print("[2/4] Embedding'ler yükleniyor...")
    z_eeg, z_periph, preds, labels = load_model_and_embeddings()

    print("[3/4] t-SNE (biraz zaman alabilir)...")
    plot_tsne(z_eeg, z_periph, labels)

    print("[4/4] Confusion matrix...")
    plot_confusion_matrix(preds, labels)

    print("[5/4] Eğitim eğrileri...")
    plot_training_curves()

    print(f"\nTüm figürler hazır → {FIG_DIR.resolve()}")