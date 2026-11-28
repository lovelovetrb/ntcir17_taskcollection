"""ベクトルを faiss の k-means で分ける (ADR-0037)。

faiss の既定値から次の 2 つを変える。

- 1 クラスタあたり 256 点までしか学習に使わない既定を外し、全件で学習する
- GPU を使うときは、見えているすべての GPU ではなく 1 枚だけを使う。使う GPU は
  `CUDA_VISIBLE_DEVICES` で選ぶ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class KMeansResult:
    """中心と割り当てと、最後の割り当てでの中心までの距離の二乗和。

    `objectives` は faiss が反復ごとに記録した目的関数の値で、試行を重ねた場合は
    試行の順に続けて並ぶ。収束したかは、各試行の最後の値の変化で確かめる。
    """

    centroids: Tensor
    labels: Tensor
    inertia: float
    objectives: tuple[float, ...]


def fit_kmeans(
    values: Tensor,
    k: int,
    *,
    seed: int,
    gpu: bool,
    iterations: int = 25,
    restarts: int = 1,
) -> KMeansResult:
    """`values` は `[件数, 次元数]`。`restarts` 回試し、目的関数が最も小さい結果を使う。"""
    points = np.ascontiguousarray(values.detach().cpu().numpy(), dtype=np.float32)
    count, dimensions = points.shape
    # faiss は SWIG で作った低水準の型しか公開せず、Python 側で包んだ train・search の
    # 呼び方を型検査が知らない
    kmeans: Any = faiss.Kmeans(
        dimensions,
        k,
        niter=iterations,
        nredo=restarts,
        seed=seed,
        gpu=1 if gpu else False,
        max_points_per_centroid=count,
        min_points_per_centroid=1,
        verbose=False,
    )
    kmeans.train(points)
    squared_distances, labels = kmeans.index.search(points, 1)
    return KMeansResult(
        centroids=torch.from_numpy(kmeans.centroids.copy()),
        labels=torch.from_numpy(labels.ravel().astype(np.int64)),
        inertia=float(squared_distances.astype(np.float64).sum()),
        objectives=tuple(float(value) for value in kmeans.obj),
    )
