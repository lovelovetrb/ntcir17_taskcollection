"""実験ごとに、回す構成の一覧を作る。

実験を足すときは、ここに関数を書いて `_BUILDERS` に登録する。実行スクリプトは
触らない。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from hidden_subspace.experiment.runner import ConfigurationPlan
from hidden_subspace.selection import SelectAll

PlanBuilder = Callable[[Sequence[int], Sequence[int]], list[ConfigurationPlan]]
"""利用できる層と次元の番号から、回す構成の一覧を作る。"""


def layer_sweep(layers: Sequence[int], dimensions: Sequence[int]) -> list[ConfigurationPlan]:
    """層を 1 つずつ使う (ADR-0006)。

    層が 1 つのときは層ごとの正規化の有無で結果が変わらないため、片方だけ回す。
    """
    del dimensions
    return [
        ConfigurationPlan(layers=(layer,), dimensions=SelectAll(), normalize_layers=False)
        for layer in layers
    ]


_BUILDERS: dict[str, PlanBuilder] = {
    "layer-sweep": layer_sweep,
}


def plan_names() -> tuple[str, ...]:
    """回せる実験の名前。"""
    return tuple(_BUILDERS)


def build_plans(
    name: str, layers: Sequence[int], dimensions: Sequence[int]
) -> list[ConfigurationPlan]:
    """名前で指定した実験の構成一覧を作る。"""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"知らない実験名: {name!r}。使えるのは {plan_names()}")
    return builder(layers, dimensions)
