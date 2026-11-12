"""層ごとの隠れ状態をファイルに保存する。

置き場所は `<collection のキャッシュ>/<model_id>/<pooling>/{documents,queries}.pt` (ADR-0035)。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

from hidden_subspace.encoding.pooling import Pooling
from hidden_subspace.states import HiddenStates

CACHE_DTYPE = torch.float16
"""保存時の精度。全 332,918 件で 6.19 GB に収まる (ADR-0015)。"""


def cache_dir(root: Path, model_id: str, pooling: Pooling) -> Path:
    """`root` はコレクションごとのキャッシュの置き場所 (`cache/<collection>`)。"""
    return root / model_id / pooling


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


def encode_or_load(path: Path, build: Callable[[], HiddenStates]) -> HiddenStates:
    """保存済みがあれば読み、なければ作って保存する。

    全 332,918 件のエンコードには 20 分以上かかるため、実験を回すたびに作り直さない。

    作った直後も保存したものを読み直して返す。そうしないと 1 度目だけ元の精度の
    まま返り、2 度目以降と結果が食い違う。
    """
    if not path.is_file():
        save_cache(build(), path)
    return load_cache(path)
