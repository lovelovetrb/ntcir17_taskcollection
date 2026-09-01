"""トークン方向に潰して、文書 1 件あたり層ごとに 1 本のベクトルにする。"""

from __future__ import annotations

from torch import Tensor


def mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """パディングを除いてトークン方向に平均する。

    `hidden_states` は `[層数, 文数, トークン数, 次元数]`、`attention_mask` は
    `[文数, トークン数]`。戻り値は `[文数, 層数, 次元数]` で、domain model が
    受け取る並びに合わせる。

    長さを揃えるために足したパディングの位置を平均に混ぜると、短い文ほど値が薄まる。
    """
    mask = attention_mask.unsqueeze(0).unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=2)
    counts = mask.sum(dim=2).clamp(min=1)
    return (summed / counts).transpose(0, 1)
