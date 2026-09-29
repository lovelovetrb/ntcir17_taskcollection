"""トピックごとに浅い側・深い側の最良の層を決め、その層で検索し直して上位を取り出す (issue #46)。

層の決め方はビューア (ADR-0020) と同じで、浅い側は 1・2 層、深い側は 11・12 層のうち
nDCG@1000 が最良のもの。同点なら番号の小さい層。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hidden_subspace.experiment.results import TopicResult
from hidden_subspace.experiment.runner import ConfigurationPlan, rank_configuration
from hidden_subspace.retrieval import Ranking
from hidden_subspace.selection import SelectAll
from hidden_subspace.states import HiddenStates

SHALLOW_LAYERS = (1, 2)
DEEP_LAYERS = (11, 12)
METRIC = "nDCG@1000"


@dataclass(frozen=True)
class TopicLayers:
    topic_id: str
    shallow: int
    deep: int

    def layer_of(self, side: str) -> int:
        return self.shallow if side == "shallow" else self.deep


def best_layers(results: Sequence[TopicResult]) -> dict[str, TopicLayers]:
    """層ごとの記録から、トピックごとに両側の最良の層を決める。"""
    by_topic: dict[str, dict[int, float]] = defaultdict(dict)
    for result in results:
        (layer,) = result.configuration.layers
        by_topic[result.topic_id][layer] = result.metrics[METRIC]

    chosen: dict[str, TopicLayers] = {}
    for topic_id, by_layer in by_topic.items():
        missing = [layer for layer in (*SHALLOW_LAYERS, *DEEP_LAYERS) if layer not in by_layer]
        if missing:
            raise ValueError(
                f"トピック {topic_id} に層 {missing} の記録がありません。"
                "layer-sweep の記録がすべての層で揃っているかを確認してください。"
            )
        chosen[topic_id] = TopicLayers(
            topic_id=topic_id,
            shallow=_best(by_layer, SHALLOW_LAYERS),
            deep=_best(by_layer, DEEP_LAYERS),
        )
    return chosen


def _best(by_layer: Mapping[int, float], layers: Sequence[int]) -> int:
    """`max` は同点のとき先に現れたものを返すため、番号順に渡して小さい層を選ぶ。"""
    return max(sorted(layers), key=lambda layer: by_layer[layer])


def rank_layers(
    layers: Sequence[int], documents: HiddenStates, queries: HiddenStates
) -> dict[int, Ranking]:
    """層ごとに 1 回だけ全トピックの順位を求める。"""
    return {
        layer: rank_configuration(
            ConfigurationPlan(layers=(layer,), dimensions=SelectAll(), normalize_layers=False),
            documents,
            queries,
        )
        for layer in sorted(set(layers))
    }


def top_documents(ranking: Ranking, topic_id: str, count: int) -> list[tuple[int, str]]:
    """(順位, 文書 ID) を順位の高い順に `count` 件。順位は 1 始まり。"""
    row = ranking.topic_ids.index(topic_id)
    positions = ranking.order[row, :count].tolist()
    return [(rank, ranking.document_ids[position]) for rank, position in enumerate(positions, 1)]
