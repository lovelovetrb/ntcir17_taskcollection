"""モデルを動かして層ごとの隠れ状態を得る処理を確かめる。

実際のモデルは読み込まず、層ごとの出力を返す偽のモデルで振る舞いだけを見る。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hidden_subspace.encoding.cache import encode_or_load, load_cache, save_cache
from hidden_subspace.encoding.pooling import mean_pool
from hidden_subspace.states import HiddenStates


def test_mean_pool_ignores_padding() -> None:
    """`[PAD]` を平均に混ぜると、短い文ほど値が薄まる。"""
    hidden = torch.tensor([[[[1.0], [3.0], [99.0]]]])  # 1 層 x 1 文 x 3 トークン
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool(hidden, mask)

    assert pooled.shape == (1, 1, 1)
    torch.testing.assert_close(pooled[0, 0, 0], torch.tensor(2.0))


def test_mean_pool_keeps_the_layer_axis() -> None:
    hidden = torch.randn(13, 4, 7, 8)  # 13 層 x 4 文 x 7 トークン x 8 次元
    mask = torch.ones(4, 7, dtype=torch.long)

    pooled = mean_pool(hidden, mask)

    assert pooled.shape == (4, 13, 8)


def test_cache_round_trip(tmp_path: Path) -> None:
    """保存して読み直したものが元と一致する。値は fp16 の精度で保たれる。"""
    states = HiddenStates.of(torch.randn(3, 5, 4, dtype=torch.float16), item_ids=("a", "b", "c"))
    path = tmp_path / "cache.pt"

    save_cache(states, path)
    loaded = load_cache(path)

    assert torch.equal(loaded.values, states.values)
    assert loaded.item_ids == states.item_ids
    assert loaded.layer_indices == states.layer_indices
    assert loaded.dimension_indices == states.dimension_indices


def test_cache_is_stored_as_half_precision(tmp_path: Path) -> None:
    """6.19 GB に収めるため fp16 に落として保持する (ADR-0015)。"""
    states = HiddenStates.of(torch.randn(3, 5, 4), item_ids=("a", "b", "c"))
    path = tmp_path / "cache.pt"

    save_cache(states, path)

    assert load_cache(path).values.dtype == torch.float16


def test_loading_a_missing_cache_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cache(tmp_path / "absent.pt")


def test_encode_or_load_builds_the_cache_when_absent(tmp_path: Path) -> None:
    calls: list[int] = []

    def build() -> HiddenStates:
        calls.append(1)
        return HiddenStates.of(torch.randn(2, 3, 4), item_ids=("a", "b"))

    path = tmp_path / "c.pt"
    first = encode_or_load(path, build)
    second = encode_or_load(path, build)

    assert calls == [1], "2 度目は保存済みのものを読む"
    assert torch.equal(first.values, second.values)
