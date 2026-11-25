"""文書と同じ平均と共分散を持つ正規分布から、比較に使う乱数データを作る (ADR-0037)。

固まりのない単峰の分布で、次元間の相関は文書と同じになる。シルエット係数をこれと比べ、
相関だけでは説明できない固まりがあるかを確かめる。
"""

from __future__ import annotations

import torch
from torch import Tensor

CHUNK = 65_536


def gaussian_like(values: Tensor, count: int, *, generator: torch.Generator) -> Tensor:
    """`values` (`[件数, 次元数]`) と同じ平均と共分散 (n で割る) を持つ `count` 件を作る。

    乱数は `generator` の置き場所で作り、`values` の置き場所に移す。返す値は float32。
    """
    mean, covariance = _moments(values)
    factor, info = torch.linalg.cholesky_ex(covariance)
    if int(info.item()) != 0:
        raise ValueError(
            "文書の共分散が正定値でないため、乱数データを作れません。"
            "値が一定の次元や、他の次元と同じ値の次元がないか確かめてください。"
        )

    dimensions = values.shape[1]
    sampled = torch.empty(count, dimensions, dtype=torch.float32, device=values.device)
    for start in range(0, count, CHUNK):
        rows = min(CHUNK, count - start)
        noise = torch.randn(
            rows, dimensions, dtype=torch.float64, generator=generator, device=generator.device
        ).to(values.device)
        sampled[start : start + rows] = (mean + noise @ factor.T).float()
    return sampled


def _moments(values: Tensor) -> tuple[Tensor, Tensor]:
    """平均と、n で割った共分散を float64 で求める。平均を引いてから掛け合わせる。"""
    count, dimensions = values.shape
    total = torch.zeros(dimensions, dtype=torch.float64, device=values.device)
    for start in range(0, count, CHUNK):
        total += values[start : start + CHUNK].double().sum(dim=0)
    mean = total / count

    products = torch.zeros(dimensions, dimensions, dtype=torch.float64, device=values.device)
    for start in range(0, count, CHUNK):
        centered = values[start : start + CHUNK].double() - mean
        products += centered.T @ centered
    return mean, products / count
