"""主成分得点と文書長の Pearson の相関係数を出す (ADR-0030)。"""

from __future__ import annotations

import torch
from torch import Tensor


def correlations(scores: Tensor, lengths: Tensor) -> Tensor:
    """`[文書数, 本数]` の得点の各列と、`[文書数]` の文書長との相関係数 `[本数]` を返す。

    float64 で計算して返す。
    """
    variables = torch.cat([scores.double().T, lengths.double()[None, :]])
    return torch.corrcoef(variables)[-1, :-1]
