"""選んだトピックが使う層で、全文書の順位を求め直す。

検索結果は実験の記録に残っていないため求め直す。記録と同じ順位にするため、実験と
同じ関数を使い、呼び出し側は実験と同じく GPU に載せた隠れ状態を渡す (ADR-0020)。
"""

from __future__ import annotations

from hidden_subspace.experiment.results import Configuration
from hidden_subspace.experiment.runner import ConfigurationPlan, rank_configuration
from hidden_subspace.retrieval import Ranking
from hidden_subspace.selection import SelectAll
from hidden_subspace.states import HiddenStates
from layer_inspector.selection import TopicSelection


def rank_selected_layers(
    selection: TopicSelection,
    documents: HiddenStates,
    queries: HiddenStates,
) -> dict[int, Ranking]:
    """層ごとに 1 回だけ検索し、層番号をキーにして返す。

    同じ層を使うトピックは、その層の順位を共有する。順位は全トピック分を並べる。
    内積はトピックごとに独立しており、絞っても各トピックの順位は変わらない。
    """
    configurations: dict[int, Configuration] = {}
    for choice in (*selection.shallow_favored, *selection.deep_favored):
        for result in (choice.shallow, choice.deep):
            (layer,) = result.configuration.layers
            configurations[layer] = result.configuration

    return {
        layer: rank_configuration(_plan_of(configurations[layer]), documents, queries)
        for layer in sorted(configurations)
    }


def _plan_of(configuration: Configuration) -> ConfigurationPlan:
    kind = configuration.dimensions.kind
    if kind != "all":
        raise ValueError(
            f"次元の選び方が {kind} の記録は検索し直せません。"
            "層ごとの記録 (layer-sweep) を渡してください。"
        )
    return ConfigurationPlan(
        layers=configuration.layers,
        dimensions=SelectAll(),
        normalize_layers=configuration.normalize_layers,
    )
