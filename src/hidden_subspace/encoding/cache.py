"""層ごとの隠れ状態をファイルに保存する。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from hidden_subspace.states import HiddenStates

CACHE_DTYPE = torch.float16
"""保存時の精度。全 332,918 件で 6.19 GB に収まる (ADR-0015)。"""


def save_cache(states: HiddenStates, path: Path) -> None:
    """隠れ状態を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "values": states.values.to(CACHE_DTYPE).cpu(),
            "item_ids": states.item_ids,
            "layer_indices": states.layer_indices,
            "dimension_indices": states.dimension_indices,
        },
        path,
    )


def load_cache(path: Path) -> HiddenStates:
    """保存した隠れ状態を読む。"""
    if not path.is_file():
        raise FileNotFoundError(f"キャッシュがない: {path}")
    stored = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    return HiddenStates(
        values=stored["values"],
        item_ids=tuple(stored["item_ids"]),
        layer_indices=tuple(stored["layer_indices"]),
        dimension_indices=tuple(stored["dimension_indices"]),
    )
