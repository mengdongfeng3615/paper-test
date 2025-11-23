"""
Utilities for subjective dissimilarity matrices and MDS/Isomap embeddings.
"""
import numpy as np
import pandas as pd
from sklearn.manifold import MDS, Isomap
from typing import Tuple

from .config import RAW_DIR


SAMPLE_ORDER = [f"Sample{i}" for i in range(1, 16)]


def load_subjective_matrix(path=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ratings, symmetrize per tester, average, and embed to 2D."""
    if path is None:
        path = RAW_DIR / "assessment_data.xlsx"
    df = pd.read_excel(path)
    matrices = []
    for tester, sub in df.groupby("tester"):
        m = (
            sub.set_index("Sample").reindex(SAMPLE_ORDER)[SAMPLE_ORDER].to_numpy(dtype=float)
        )
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        matrices.append(m)
    avg = np.mean(np.stack(matrices, axis=0), axis=0)
    avg = avg - np.min(avg)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    mds_coords = mds.fit_transform(avg)
    isomap = Isomap(n_neighbors=5, n_components=2, metric="precomputed")
    iso_coords = isomap.fit_transform(avg)
    return avg, mds_coords, iso_coords
