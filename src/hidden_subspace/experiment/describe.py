"""選び方から、結果に残す表現を導く。

呼び出し側が選び方とその表現を別々に渡すと、片方だけ変えたときに食い違う。
選び方そのものから導くことで食い違いを防ぐ。
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from hidden_subspace.experiment.results import DimensionChoice
from hidden_subspace.selection import Selector

_KIND_PREFIX = "Select"


def describe_dimensions(selector: Selector) -> DimensionChoice:
    """次元の選び方を、記録用の表現にする。

    種類は型の名前から導く。`SelectRandom` なら `random`。引数は dataclass の
    フィールドをそのまま使う。
    """
    if isinstance(selector, type) or not is_dataclass(selector):
        raise TypeError(f"選び方は dataclass の値である必要がある: {type(selector).__name__}")
    kind = type(selector).__name__.removeprefix(_KIND_PREFIX).lower()
    parameters = {f.name: getattr(selector, f.name) for f in fields(selector)}
    return DimensionChoice(kind=kind, parameters=parameters)
