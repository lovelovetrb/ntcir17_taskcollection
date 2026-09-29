"""トピック別の値どうしの順位相関と、その信頼区間。scipy を使わず numpy で計算する。"""

from __future__ import annotations

import numpy as np


def ranks(values: np.ndarray) -> np.ndarray:
    """1 始まりの順位。同点は順位の平均にする (nDCG は 0 で同点になりやすい)。"""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and ordered[end + 1] == ordered[start]:
            end += 1
        result[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def bootstrap_interval(
    x: np.ndarray, y: np.ndarray, *, resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """トピックを復元抽出して相関を計算し直し、その分布の両端を返す。"""
    generator = np.random.default_rng(seed)
    count = len(x)
    draws = np.empty(resamples)
    for i in range(resamples):
        chosen = generator.integers(0, count, size=count)
        draws[i] = spearman(x[chosen], y[chosen])
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))
