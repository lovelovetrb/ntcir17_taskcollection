"""選んだ点について、全件までの距離からシルエット係数を求める (ADR-0037)。

点 i のシルエット係数は (b - a) / max(a, b)。a は同じクラスタの他の点までの平均距離、
b は他のクラスタのうち平均距離が最も小さいものまでの平均距離。1 点だけのクラスタの点は
scikit-learn と同じく 0 にする。

全件どうしの距離は件数の 2 乗になり計算しきれないため、点は選んだものだけにする。
それぞれの点の値は全件までの距離で求めるので、選んだ点については厳密な値になる。
"""

from __future__ import annotations

import torch
from torch import Tensor

BATCH = 1_024
CHUNK = 65_536


def sampled_silhouette(
    values: Tensor,
    labels: Tensor,
    chosen: Tensor,
    *,
    k: int,
    batch: int = BATCH,
    chunk: int = CHUNK,
) -> Tensor:
    """`values` は `[件数, 次元数]`、`labels` は全件のクラスタ番号、`chosen` は選んだ点の番号。

    距離は float64 で求める。返す値は選んだ点の順に並べた float64 の CPU のテンソル。
    """
    device = values.device
    labels = labels.to(device)
    chosen = chosen.to(device)
    counts = torch.bincount(labels, minlength=k).double()
    results: list[Tensor] = []

    for start in range(0, chosen.shape[0], batch):
        rows = chosen[start : start + batch]
        sums = _distance_sums(values, labels, rows, k=k, chunk=chunk)
        own = labels[rows]
        own_counts = counts[own]

        inner = sums.gather(1, own.unsqueeze(1)).squeeze(1) / (own_counts - 1).clamp_min(1)
        means = sums / counts
        means[:, counts == 0] = torch.inf
        means.scatter_(1, own.unsqueeze(1), torch.inf)
        nearest = means.min(dim=1).values

        silhouette = (nearest - inner) / torch.maximum(inner, nearest)
        results.append(torch.where(own_counts > 1, silhouette, 0.0).cpu())

    return torch.cat(results)


def _distance_sums(values: Tensor, labels: Tensor, rows: Tensor, *, k: int, chunk: int) -> Tensor:
    """選んだ点から全件までの距離を、クラスタごとに足した `[選んだ点, k]`。

    距離は |x|² + |y|² - 2x·y から求める。自分自身までの距離は丸め誤差で 0 にならない
    ため、0 に置き直す。
    """
    chosen_values = values[rows].double()
    chosen_norms = (chosen_values**2).sum(dim=1, keepdim=True)
    sums = torch.zeros(rows.shape[0], k, dtype=torch.float64, device=values.device)

    for start in range(0, values.shape[0], chunk):
        block = values[start : start + chunk].double()
        squared = chosen_norms + (block**2).sum(dim=1) - 2 * chosen_values @ block.T
        distances = squared.clamp_min(0).sqrt()
        inside = (rows >= start) & (rows < start + block.shape[0])
        distances[inside.nonzero().squeeze(1), rows[inside] - start] = 0.0
        sums.index_add_(1, labels[start : start + chunk], distances)

    return sums
