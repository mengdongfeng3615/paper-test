"""
Subjective matrix loading and 4D MDS embedding.
"""
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from typing import Tuple

from .config import RAW_DIR

SAMPLE_ORDER = [f"Sample{i}" for i in range(1, 16)]


def load_mds_4d(path=None) -> Tuple[np.ndarray, np.ndarray]:
    if path is None:
        path = RAW_DIR / "assessment_data.xlsx"
    df = pd.read_excel(path)
    mats = []
    for _, sub in df.groupby("tester"):
        m = sub.set_index("Sample").reindex(SAMPLE_ORDER)[SAMPLE_ORDER].to_numpy(dtype=float)
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        mats.append(m)
    avg = np.mean(np.stack(mats, axis=0), axis=0)
    avg = avg - np.min(avg)
    mds = MDS(n_components=4, dissimilarity="precomputed", random_state=0)
    coords = mds.fit_transform(avg)
    return avg, coords
