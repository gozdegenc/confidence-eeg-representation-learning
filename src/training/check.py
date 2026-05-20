import numpy as np
from pathlib import Path

DATA_DIR = Path('../../data/processed_cosubio')
behavioral = np.load(DATA_DIR / 'behavioral.npy')
labels     = np.load(DATA_DIR / 'labels.npy')

n = len(behavioral)
n_sub = 34
epp = n // n_sub

print('Epoch sayisi:', n)
print('Subject basina:', epp)
print()
for sid in [0, 1, 2, 33]:
    start = sid * epp
    v = behavioral[start]
    print(f'Subject {sid+1:02d} behavioral[0]: {v[:3].round(3)}')