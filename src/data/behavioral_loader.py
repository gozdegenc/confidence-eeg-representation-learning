"""
CoSuBio Behavioral Performance + Self-Esteem Loader
game_neutral/positive/negative.csv + Gender_and_SelfEsteem.xlsx
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.preprocessing import StandardScaler


GAME_COLS = ['attention','flexibility','memory',
             'problem_solving','speed','puzzle_total','puzzle_dist']
SE_COLS   = ['Rosenberg Self-Esteem Scale Score',
             'eneral Self-Efficacy Scale Score']
PHASE_MAP = {0: 'game_neutral.csv',
             1: 'game_positive.csv',
             2: 'game_negative.csv'}


def load_behavioral(data_dir: Union[str, Path]) -> dict:
    """
    Her subject × faz kombinasyonu için behavioral vektör döndürür.

    Döndürür:
      behavioral_matrix: (34, 3, 9) — subject × faz × özellik
      feature_names: list[str]
    """
    data_dir = Path(data_dir)
    stats_dir = data_dir / 'statistics'
    info_dir  = data_dir / 'Participant Information'

    # ── Game skorları ──────────────────────────────────
    game_data = {}
    for phase_id, fname in PHASE_MAP.items():
        df = pd.read_csv(stats_dir / fname, sep=';')
        df = df.set_index('sid').sort_index()
        game_data[phase_id] = df[GAME_COLS]

    # ── Self-esteem ────────────────────────────────────
    se_df = pd.read_excel(info_dir / 'Gender_and_SelfEsteem_Results.xlsx')
    se_df = se_df.set_index('sid').sort_index()

    subjects = sorted(se_df.index.tolist())
    n_subjects = len(subjects)   # 34

    # ── Birleştir: (34, 3, 9) ─────────────────────────
    # 7 game skoru + 2 self-esteem skoru = 9 özellik
    behavioral = np.zeros((n_subjects, 3, 9), dtype=np.float32)

    for si, sid in enumerate(subjects):
        for phase_id in range(3):
            game_scores = np.asarray(game_data[phase_id].loc[sid, GAME_COLS], dtype=np.float32)
            se_scores   = np.asarray(se_df.loc[sid, SE_COLS], dtype=np.float32)
            behavioral[si, phase_id] = np.concatenate([
                game_scores, se_scores
            ])

    # ── Normalize et ───────────────────────────────────
    # Her özellik boyutu için tüm subject+faz kombinasyonları üzerinde
    flat = behavioral.reshape(-1, 9)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    behavioral = flat_scaled.reshape(n_subjects, 3, 9)

    feature_names = GAME_COLS + SE_COLS

    return {
        'behavioral': behavioral.astype(np.float32),
        'subjects': subjects,
        'feature_names': feature_names,
    }


def assign_behavioral_to_epochs(behavioral_matrix: np.ndarray,
                                 subject_ids: np.ndarray,
                                 phase_labels: np.ndarray) -> np.ndarray:
    """
    Her epoch'a subject ve faz bazinda behavioral vektor ata.
    behavioral_matrix: (34, 3, 9) — subject x faz x ozellik
    subject_ids: (N,) — 1-indexed
    phase_labels: (N,) — 0/1/2
    """
    n_epochs = len(subject_ids)
    result = np.zeros((n_epochs, 9), dtype=np.float32)

    unique_subjects = np.unique(subject_ids)

    for i in range(n_epochs):
        sid   = int(subject_ids[i])
        phase = int(phase_labels[i])

        # subject_ids 1-indexed, behavioral_matrix 0-indexed
        # unique_subjects icindeki pozisyonu bul
        sid_idx = np.where(unique_subjects == sid)[0][0]

        if sid_idx < behavioral_matrix.shape[0] and phase < behavioral_matrix.shape[1]:
            result[i] = behavioral_matrix[sid_idx, phase]
        # else: sifir kalir

    return result