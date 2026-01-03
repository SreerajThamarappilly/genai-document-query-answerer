# src/utils/visualize.py
"""
Visualization utilities.

Provides a function to project high‑dimensional embeddings to 2D via PCA and plot them.  
Useful to inspect how sections cluster.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embeddings(embeddings: np.ndarray, labels: Optional[List[str]] = None):
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be non‑empty 2D array")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals, y_vals = reduced[:, 0], reduced[:, 1]
    if labels:
        unique_labels = list(dict.fromkeys(labels))
        palette = plt.cm.get_cmap('tab10', len(unique_labels))
        for idx, label in enumerate(unique_labels):
            xs = [x for x, lbl in zip(x_vals, labels) if lbl == label]
            ys = [y for y, lbl in zip(y_vals, labels) if lbl == label]
            ax.scatter(xs, ys, label=str(label), alpha=0.7, color=palette(idx))
        ax.legend(title="Section Type/Page", fontsize='small')
    else:
        ax.scatter(x_vals, y_vals, alpha=0.7)
    ax.set_title("Embedding Projection (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig
