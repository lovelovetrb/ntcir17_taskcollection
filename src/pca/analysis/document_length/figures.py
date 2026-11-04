"""主成分得点と文書長の散布図を描く (ADR-0030)。

図の文字は英語にする。環境の日本語フォントに依らずに同じ図が出るようにするため。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

POINT_COLOR = "#2a78d6"
SAMPLE_SIZE = 10_000
SEED = 0


def save_scatter_grid(
    lengths: np.ndarray,
    scores: np.ndarray,
    pcs: Sequence[int],
    r2: Sequence[float],
    *,
    title: str,
    path: Path,
) -> None:
    """主成分ごとの散布図を 3 行 3 列に並べ、右上にグループの全文書で計算した r² を載せる。

    `scores[:, i]` は `pcs[i]` 本目の主成分の得点。点は無作為に選んだ `SAMPLE_SIZE` 件だけ描く。
    """
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(lengths), size=min(SAMPLE_SIZE, len(lengths)), replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for ax, pc, column, value in zip(axes.flat, pcs, scores.T, r2, strict=True):
        ax.scatter(lengths[sample], column[sample], s=2, alpha=0.3, color=POINT_COLOR, linewidths=0)
        ax.set_title(f"PC{pc}")
        ax.text(0.97, 0.95, f"r² = {value:.3f}", transform=ax.transAxes, ha="right", va="top")
        ax.set_xlabel("document length (tokens)")
        ax.set_ylabel("PC score")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
