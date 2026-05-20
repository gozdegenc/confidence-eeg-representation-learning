"""
Gerçek veri gelmeden pipeline'ı test etmek için
DEAP ile aynı shape'te rastgele veri üretir.
"""
import numpy as np
from pathlib import Path

def generate_mock_deap(n_subjects=4, n_trials=40,
                       n_eeg_channels=32, n_periph_channels=8,
                       n_samples=8064, save_dir="../../data/raw/deap"):
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for sid in range(1, n_subjects + 1):
        # Gerçek DEAP ile aynı yapı
        data = {
            'data':   np.random.randn(n_trials,
                                      n_eeg_channels + n_periph_channels,
                                      n_samples).astype(np.float32),
            'labels': np.random.uniform(1, 9,
                                        size=(n_trials, 4)).astype(np.float32)
        }
        import pickle
        with open(f"{save_dir}/s{sid:02d}.dat", 'wb') as f:
            pickle.dump(data, f)
        print(f"Mock subject {sid:02d} oluşturuldu")
    
    print("Tüm mock veri hazır — pipeline test edilebilir.")

if __name__ == "__main__":
    generate_mock_deap()