"""トークン方向に潰して、文書 1 件あたり層ごとに 1 本のベクトルにする。

プーリングは `mean` と `cls` から選ぶ (ADR-0035)。`cls` では層 0 (埋め込み層) を除く。
埋め込み層の先頭トークンは全文書で同じベクトルになり、分散が 0 で主成分も推定できず、
検索しても全文書が同点になるため (ADR-0034)。
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor

Pooling = Literal["mean", "cls"]
MEAN: Pooling = "mean"
CLS: Pooling = "cls"
POOLINGS: tuple[Pooling, ...] = (MEAN, CLS)


def mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """`[PAD]` トークンを除いてトークン方向に平均する。

    `hidden_states` は `[層数, 文数, トークン数, 次元数]`、`attention_mask` は
    `[文数, トークン数]`。戻り値は `[文数, 層数, 次元数]` で、domain model が
    受け取る並びに合わせる。

    バッチ内で長さを揃えるために足された `[PAD]` を平均に混ぜると、短い文ほど値が
    薄まる。`attention_mask` が 0 の位置がそれにあたる。
    """
    mask = attention_mask.unsqueeze(0).unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=2)
    counts = mask.sum(dim=2).clamp(min=1)
    return (summed / counts).transpose(0, 1)


def cls_pool(hidden_states: Tensor) -> Tensor:
    """各層の先頭トークンの隠れ状態を取る。並びは `mean_pool` と同じ。"""
    return hidden_states[:, :, 0, :].transpose(0, 1)


def first_layer(pooling: Pooling) -> int:
    """キャッシュに残す最初の層の番号。"""
    return 1 if pooling == CLS else 0


def pool(hidden_states: Tensor, attention_mask: Tensor, pooling: Pooling) -> Tensor:
    """`pooling` で選んだ方法で潰し、`first_layer` より前の層を除く。"""
    kept = hidden_states[first_layer(pooling) :]
    if pooling == CLS:
        return cls_pool(kept)
    return mean_pool(kept, attention_mask)
