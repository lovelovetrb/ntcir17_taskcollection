"""無作為に選ぶ選び方。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectRandom:
    """無作為に `count` 個残す。

    `seed` が同じなら、同じ番号の列に対して常に同じ結果を返す。種を変えて何本も
    引き、性能の散らばりを見る実験のために必要となる。
    """

    count: int
    seed: int

    def __call__(self, available: Sequence[int]) -> tuple[int, ...]:
        if self.count > len(available):
            raise ValueError(
                f"""選択個数が母集団を超えています。
                指定選択数: {self.count} : 母集団次元数{len(available)}"""
            )
        chosen = random.Random(self.seed).sample(list(available), self.count)
        return tuple(sorted(chosen))
