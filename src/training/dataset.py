
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class EEGConfidenceDataset(Dataset):
    """
    İşlenmiş .npy dosyalarını (eeg, peripheral, labels) PyTorch tensörlerine yükler.
    Her __getitem__ çağrısı bir epoch için EEG, periferik sinyal ve etiketi döndürür.
    """

    def __init__(self, data_dir: str | Path, split: str = 'train',
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42):
        """
        split: 'train', 'val', veya 'test' — hangi bölüm yükleneceğini belirler
        """
        data_dir = Path(data_dir)

        eeg    = np.load(data_dir / 'eeg.npy').astype(np.float32)
        periph = np.load(data_dir / 'peripheral.npy').astype(np.float32)
        labels = np.load(data_dir / 'labels.npy')

        # CoSuBio etiketleri zaten 0/1/2 (nötr/pozitif/negatif); doğrudan kullan
        y = labels.astype(np.int64)

        # Sabit tohum ile karıştır; her çalıştırmada aynı bölünme elde edilir
        n = len(eeg)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)

        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)

        # Test önce, sonra val, geri kalanı train — sıra önemli (aynı idx üzerinde)
        if split == 'test':
            idx = idx[:n_test]
        elif split == 'val':
            idx = idx[n_test:n_test + n_val]
        else:
            idx = idx[n_test + n_val:]

        self.eeg    = torch.from_numpy(eeg[idx])
        self.periph = torch.from_numpy(periph[idx])
        self.labels = torch.from_numpy(y[idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'eeg':        self.eeg[idx],
            'peripheral': self.periph[idx],
            'label':      self.labels[idx],
        }