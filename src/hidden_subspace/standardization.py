"""隠れ状態を層・次元ごとに z スコアに直す (ADR-0037)。

平均と標準偏差は文書から求め、問いにも同じ値を当てる。どちらに当てるかは呼び出し側が決める。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from hidden_subspace.states import HiddenStates

CHUNK_ELEMENTS = 65_536 * 1_024
"""1 回に float64 に直す要素数の上限 (512 MiB)。行数は層数と次元数に応じて決める。

NTCIR-2 の xlm-roberta-large の文書を全 25 層まとめて直すと 140 GiB を超える。
"""


@dataclass(frozen=True)
class Standardization:
    """`[層数, 次元数]` の平均と標準偏差と、それが元のモデルのどの層・次元か。"""

    mean: Tensor
    std: Tensor
    layer_indices: tuple[int, ...]
    dimension_indices: tuple[int, ...]

    def apply(self, states: HiddenStates) -> HiddenStates:
        """層や次元を絞った状態にも、元の番号で対応する値を当てる。"""
        layers = _positions(states.layer_indices, self.layer_indices, "層")
        dimensions = _positions(states.dimension_indices, self.dimension_indices, "次元")
        mean = self.mean[layers][:, dimensions].to(states.values.device)
        std = self.std[layers][:, dimensions].to(states.values.device)
        values = torch.empty(states.values.shape, dtype=torch.float32, device=states.values.device)
        for start, chunk in _chunks(states.values):
            values[start : start + chunk.shape[0]] = ((chunk - mean) / std).float()
        return replace(states, values=values)


def fit_standardization(documents: HiddenStates) -> Standardization:
    """標準偏差は n で割る。精度を保つため float64 で、平均を引いてから二乗する。"""
    count = documents.item_count
    if count == 0:
        raise ValueError(
            "文書が 1 件もありません。平均と標準偏差を求められません。"
            "文書の隠れ状態を渡してください。"
        )
    total = torch.zeros(documents.layer_count, documents.dimension_count, dtype=torch.float64)
    total = total.to(documents.values.device)
    for _, chunk in _chunks(documents.values):
        total += chunk.sum(dim=0)
    mean = total / count

    squared = torch.zeros_like(total)
    for _, chunk in _chunks(documents.values):
        squared += ((chunk - mean) ** 2).sum(dim=0)
    std = (squared / count).sqrt()

    zero = (std == 0).nonzero().tolist()
    if zero:
        pairs = [
            (documents.layer_indices[layer], documents.dimension_indices[dimension])
            for layer, dimension in zero
        ]
        raise ValueError(
            f"標準偏差が 0 の (層, 次元) があるため、z スコアに直せません: {pairs}。"
            "文書の隠れ状態を確かめてください。"
        )

    return Standardization(
        mean=mean.cpu(),
        std=std.cpu(),
        layer_indices=documents.layer_indices,
        dimension_indices=documents.dimension_indices,
    )


def _chunks(values: Tensor) -> Iterator[tuple[int, Tensor]]:
    """先頭の行番号と、float64 に直した行の塊を返す。"""
    _, layers, dimensions = values.shape
    rows = max(CHUNK_ELEMENTS // (layers * dimensions), 1)
    for start in range(0, values.shape[0], rows):
        yield start, values[start : start + rows].double()


def _positions(wanted: tuple[int, ...], available: tuple[int, ...], axis_name: str) -> Tensor:
    position_of = {number: position for position, number in enumerate(available)}
    unknown = [number for number in wanted if number not in position_of]
    if unknown:
        raise ValueError(
            f"平均と標準偏差を求めていない{axis_name}があります: {unknown}。"
            "同じ隠れ状態から求め直してください。"
        )
    return torch.tensor([position_of[number] for number in wanted], dtype=torch.long)
