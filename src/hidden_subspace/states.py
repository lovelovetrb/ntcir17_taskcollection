"""層ごとの隠れ状態を表す domain model。

層を絞る操作と次元を絞る操作は、いずれも同じ構造を返す独立した操作として
定義する。両者は可換であり、絞り込んだ後も元のモデルにおける層番号と次元番号を
保持する (ADR-0016)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class HiddenStates:
    """`[要素数, 層数, 次元数]` の隠れ状態と、それが何であるかの対応。

    `layer_indices` と `dimension_indices` は元のモデルにおける番号であり、
    絞り込みを重ねても引き継がれる。選択の結果を「128 次元」ではなく
    「第 42・117・305 次元」として読むために必要となる。
    """

    values: Tensor
    item_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    dimension_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"values は 3 階のテンソルである必要がある: {self.values.shape}")
        rows, layers, dimensions = self.values.shape
        if len(self.item_ids) != rows:
            raise ValueError(f"item_ids の数 {len(self.item_ids)} が行数 {rows} と一致しない")
        if len(self.layer_indices) != layers:
            raise ValueError(
                f"layer_indices の数 {len(self.layer_indices)} が層数 {layers} と一致しない"
            )
        if len(self.dimension_indices) != dimensions:
            raise ValueError(
                f"dimension_indices の数 {len(self.dimension_indices)} が"
                f"次元数 {dimensions} と一致しない"
            )

    @classmethod
    def of(cls, values: Tensor, item_ids: Sequence[str]) -> HiddenStates:
        """絞り込みを受けていない状態を作る。層番号と次元番号は 0 から振る。"""
        if values.ndim != 3:
            raise ValueError(f"values は 3 階のテンソルである必要がある: {values.shape}")
        _, layers, dimensions = values.shape
        return cls(
            values=values,
            item_ids=tuple(item_ids),
            layer_indices=tuple(range(layers)),
            dimension_indices=tuple(range(dimensions)),
        )

    @property
    def item_count(self) -> int:
        return self.values.shape[0]

    @property
    def layer_count(self) -> int:
        return self.values.shape[1]

    @property
    def dimension_count(self) -> int:
        return self.values.shape[2]

    def to(self, device: str | torch.device) -> HiddenStates:
        """テンソルを指定した置き場所へ移す。何であるかは変わらない。"""
        return replace(self, values=self.values.to(device))

    def select_layers(self, layers: Sequence[int]) -> HiddenStates:
        """指定した層だけを残す。次元は変わらない。"""
        positions = _positions_of(layers, self.layer_indices, "層", self.values.device)
        return replace(
            self,
            values=self.values.index_select(1, positions),
            layer_indices=tuple(layers),
        )

    def select_dimensions(self, dimensions: Sequence[int]) -> HiddenStates:
        """指定した次元だけを残す。層は変わらない。"""
        positions = _positions_of(dimensions, self.dimension_indices, "次元", self.values.device)
        return replace(
            self,
            values=self.values.index_select(2, positions),
            dimension_indices=tuple(dimensions),
        )


def _positions_of(
    wanted: Sequence[int],
    available: tuple[int, ...],
    axis_name: str,
    device: torch.device,
) -> Tensor:
    """元の番号の列を、いま保持しているテンソル上の位置に変換する。

    添字は値と同じ置き場所に作る。揃っていないと `index_select` が拒否する。
    """
    position_of = {number: position for position, number in enumerate(available)}
    unknown = [number for number in wanted if number not in position_of]
    if unknown:
        raise ValueError(f"保持していない{axis_name}が指定された: {unknown}")
    return torch.tensor([position_of[number] for number in wanted], dtype=torch.long, device=device)
