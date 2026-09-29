"""トピック別の値どうしの順位相関と、その信頼区間。計算は scipy に任せる。"""

from __future__ import annotations

import numpy as np
from scipy import stats


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """同点は平均順位で扱われる (nDCG は 0 で同点になりやすい)。"""
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_interval(
    x: np.ndarray, y: np.ndarray, *, resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """トピックを対にしたまま復元抽出して相関を計算し直し、分布の両端 (percentile 法) を返す。"""
    result = stats.bootstrap(
        (x, y),
        statistic=spearman,
        paired=True,
        vectorized=False,
        n_resamples=resamples,
        confidence_level=level,
        method="percentile",
        rng=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)
