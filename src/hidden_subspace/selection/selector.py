"""選び方の共通の形。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Selector(Protocol):
    """利用可能な番号の列から、残す番号の列を選ぶ。

    層か次元かを知る必要がないため、同じ実装を両方の軸に使える。
    """

    def __call__(self, available: Sequence[int]) -> tuple[int, ...]: ...
