"""モデルを動かして層ごとの隠れ状態を得る処理を確かめる。

実際のモデルは読み込まず、層ごとの出力を返す偽のモデルで振る舞いだけを見る。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import BatchEncoding

from hidden_subspace.encoding.cache import cache_dir, encode_or_load, load_cache, save_cache
from hidden_subspace.encoding.encoder import Encoder
from hidden_subspace.encoding.pooling import Pooling, cls_pool, mean_pool, pool
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


def test_cls_pool_takes_the_first_token_of_each_layer() -> None:
    hidden = torch.arange(2 * 3 * 4 * 1, dtype=torch.float).reshape(2, 3, 4, 1)  # 2 層 x 3 文

    pooled = cls_pool(hidden)

    assert pooled.shape == (3, 2, 1)
    torch.testing.assert_close(pooled[:, :, 0], hidden[:, :, 0, 0].T)


def test_cls_pooling_leaves_out_the_embedding_layer() -> None:
    """CLS の層 0 は全文書で同じベクトルになり、分散が 0 になる (ADR-0034)。"""
    hidden = torch.randn(13, 4, 7, 8)
    mask = torch.ones(4, 7, dtype=torch.long)

    torch.testing.assert_close(pool(hidden, mask, "cls"), cls_pool(hidden[1:]))
    torch.testing.assert_close(pool(hidden, mask, "mean"), mean_pool(hidden, mask))


class FakeTokenizer:
    """1 文を 3 トークンにする。"""

    def __call__(self, texts: list[str], **_: Any) -> BatchEncoding:
        count = len(texts)
        return BatchEncoding(
            {
                "input_ids": torch.zeros(count, 3, dtype=torch.long),
                "attention_mask": torch.ones(count, 3, dtype=torch.long),
            }
        )


class FakeModel:
    """埋め込み層と 2 層の、合わせて 3 つの出力を返す。値は層の番号にする。"""

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        shape = (*input_ids.shape, 4)
        return SimpleNamespace(hidden_states=tuple(torch.full(shape, float(i)) for i in range(3)))


def fake_encoder(pooling: Pooling) -> Encoder:
    """モデルを読み込まずに作る。"""
    encoder = Encoder.__new__(Encoder)
    encoder.model_id = "fake"
    encoder.pooling = pooling
    encoder.device = "cpu"
    encoder.tokenizer = FakeTokenizer()  # type: ignore[assignment]
    encoder.model = FakeModel()  # type: ignore[assignment]
    return encoder


def test_a_cls_encoding_starts_at_layer_one() -> None:
    """層を除いても、番号は元のモデルの層番号のまま持つ (ADR-0016)。"""
    states = fake_encoder("cls").encode(["a", "b"], ["d0", "d1"])

    assert states.layer_indices == (1, 2)
    torch.testing.assert_close(states.values[0, :, 0], torch.tensor([1.0, 2.0]).half())


def test_a_mean_encoding_keeps_every_layer() -> None:
    states = fake_encoder("mean").encode(["a", "b"], ["d0", "d1"])

    assert states.layer_indices == (0, 1, 2)
    assert states.item_ids == ("d0", "d1")


def test_the_cache_is_kept_apart_for_each_pooling(tmp_path: Path) -> None:
    """同じモデルの CLS と mean は別のベクトルになる (ADR-0035)。"""
    mean = cache_dir(tmp_path, "cl-tohoku/bert-base-japanese-v3", "mean")
    cls = cache_dir(tmp_path, "cl-tohoku/bert-base-japanese-v3", "cls")

    assert mean == tmp_path / "cl-tohoku" / "bert-base-japanese-v3" / "mean"
    assert cls != mean
