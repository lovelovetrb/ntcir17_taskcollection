"""浅い層と深い層で得意なトピックを選ぶ。

層ごとの性能の記録だけで決まるため、検索し直す必要はない。どの層を浅い層・深い層と
みなすかは呼び出し側が渡す。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hidden_subspace.experiment.results import TopicResult


@dataclass(frozen=True)
class TopicChoice:
    """観察するトピックと、浅い層・深い層それぞれで指標が最良だった層の記録。

    不得意な側の記録も持つ。文書の表では、同じ文書を反対側の層で何位に置いたかも並べる。
    """

    topic_id: str
    shallow: TopicResult
    deep: TopicResult
    margin: float
    """浅い層の値から深い層の値を引いた差。正なら浅い層が、負なら深い層が優位。"""


@dataclass(frozen=True)
class TopicSelection:
    shallow_favored: list[TopicChoice]
    deep_favored: list[TopicChoice]


def select_topics(
    results: Sequence[TopicResult],
    metric: str,
    shallow_layers: Sequence[int],
    deep_layers: Sequence[int],
    count: int,
) -> TopicSelection:
    """差の符号で群に分け、それぞれ差の大きい順に `count` 件ずつ選ぶ。

    差で並べて両端から取ると、件数を増やしたときに反対側が優位なトピックまで入る。
    先に符号で分ければ、群の名前と中身が食い違わない。差が 0 のトピックはどちらも
    優位ではないため、どちらの群にも入れない。
    """
    by_topic: dict[str, dict[int, TopicResult]] = defaultdict(dict)
    for result in results:
        (layer,) = result.configuration.layers
        by_topic[result.topic_id][layer] = result

    choices = [
        _choose(topic_id, by_layer, metric, shallow_layers, deep_layers)
        for topic_id, by_layer in by_topic.items()
    ]
    shallow_favored = sorted(
        (choice for choice in choices if choice.margin > 0),
        key=lambda choice: choice.margin,
        reverse=True,
    )
    deep_favored = sorted(
        (choice for choice in choices if choice.margin < 0),
        key=lambda choice: choice.margin,
    )
    return TopicSelection(
        shallow_favored=shallow_favored[:count],
        deep_favored=deep_favored[:count],
    )


def _choose(
    topic_id: str,
    by_layer: Mapping[int, TopicResult],
    metric: str,
    shallow_layers: Sequence[int],
    deep_layers: Sequence[int],
) -> TopicChoice:
    missing = [layer for layer in (*shallow_layers, *deep_layers) if layer not in by_layer]
    if missing:
        raise ValueError(
            f"トピック {topic_id} に層 {missing} の記録がありません。"
            "layer-sweep の記録がすべての層で揃っているかを確認してください。"
        )
    shallow = _best(by_layer, shallow_layers, metric)
    deep = _best(by_layer, deep_layers, metric)
    return TopicChoice(
        topic_id=topic_id,
        shallow=shallow,
        deep=deep,
        margin=shallow.metrics[metric] - deep.metrics[metric],
    )


def _best(by_layer: Mapping[int, TopicResult], layers: Sequence[int], metric: str) -> TopicResult:
    """同点なら番号の小さい層を選ぶ。`max` は同点のとき先に現れたものを返すため、番号順に渡す。"""
    best = max(sorted(layers), key=lambda layer: by_layer[layer].metrics[metric])
    return by_layer[best]
