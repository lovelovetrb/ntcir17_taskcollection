"""すべて残す選び方。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectAll:
    """すべて残す。"""

    def __call__(self, available: Sequence[int]) -> tuple[int, ...]:
        return tuple(available)
