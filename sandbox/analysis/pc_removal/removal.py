"""文書集合の平均と上位主成分を推定し、それをベクトルから除く。

主成分は文書を中心化してから求める。中心化しないと第 1 主成分が平均の向きそのものに
なり、「平均を引く」と「主成分を除く」が 1 本目で混ざって k の意味がずれる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CommonComponents:
    """文書集合の平均と、分散の大きい順に並べた主成分。"""

    mean: Tensor
    """`[次元数]`。"""
    directions: Tensor
    """`[次元数, 本数]`。各列が長さ 1 で、互いに直交する。"""

    def remove(self, values: Tensor, count: int) -> Tensor:
        """平均を引き、上位 `count` 本の主成分の向きの成分を除く。

        クエリにも文書の平均と主成分をそのまま当てる。別の平均を引くと、原点の
        ずれた空間どうしを内積で比べることになる。
        """
        if count > self.directions.shape[1]:
            raise ValueError(
                f"推定した主成分は {self.directions.shape[1]} 本です。"
                f"{count} 本を除くには、推定する本数を増やしてください。"
            )
        centered = values.to(self.mean.dtype) - self.mean
        top = self.directions[:, :count]
        return centered - centered @ top @ top.T


def estimate_common_components(documents: Tensor, count: int) -> CommonComponents:
    """`[文書数, 次元数]` から平均と上位 `count` 本の主成分を求める。

    float64 で計算する。キャッシュは float16 で、33 万件の共分散を半精度や単精度で
    足し込むと下位の固有ベクトルの向きが崩れる。
    """
    values = documents.double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    _, vectors = torch.linalg.eigh(covariance)
    return CommonComponents(mean=mean, directions=vectors.flip(1)[:, :count])
