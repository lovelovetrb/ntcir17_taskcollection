"""浅い層と深い層で得意なトピックを選ぶ処理を確かめる。"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from hidden_subspace.experiment.results import Configuration, DimensionChoice, TopicResult
from layer_inspector.selection import select_topics

METRIC = "nDCG@1000"
SHALLOW = (1, 2)
DEEP = (11, 12)


def result_of(topic_id: str, layer: int, score: float) -> TopicResult:
    configuration = Configuration(
        model_id="cl-tohoku/bert-base-japanese-v3",
        layers=(layer,),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
    )
    return TopicResult("layer-sweep", configuration, topic_id, {METRIC: score})


def results_of(scores: Mapping[str, Mapping[int, float]]) -> list[TopicResult]:
    """トピック ID から「層番号 → 指標の値」への対応を、記録の一覧にする。"""
    return [
        result_of(topic_id, layer, score)
        for topic_id, by_layer in scores.items()
        for layer, score in by_layer.items()
    ]


def test_picks_the_best_layer_on_each_side() -> None:
    results = results_of({"0001": {1: 0.2, 2: 0.5, 11: 0.4, 12: 0.1}})

    selection = select_topics(results, METRIC, SHALLOW, DEEP, count=5)

    (choice,) = selection.shallow_favored
    assert choice.shallow == result_of("0001", 2, 0.5)
    assert choice.deep == result_of("0001", 11, 0.4)
    assert choice.margin == pytest.approx(0.1)


def test_a_tie_goes_to_the_smaller_layer_number() -> None:
    """同点の層を並び順に任せると、記録の順が変わっただけで選ばれる層が変わる。"""
    results = results_of({"0001": {2: 0.5, 1: 0.5, 12: 0.1, 11: 0.1}})

    selection = select_topics(results, METRIC, SHALLOW, DEEP, count=5)

    (choice,) = selection.shallow_favored
    assert choice.shallow.configuration.layers == (1,)
    assert choice.deep.configuration.layers == (11,)


def test_uses_only_the_given_layers() -> None:
    results = results_of({"0001": {0: 0.1, 3: 0.9, 5: 0.4}})

    selection = select_topics(results, METRIC, (0,), (5,), count=5)

    (choice,) = selection.deep_favored
    assert choice.shallow.configuration.layers == (0,)
    assert choice.deep.configuration.layers == (5,)


def test_splits_topics_by_the_sign_of_the_margin() -> None:
    """差が 0 のトピックはどちらも優位ではないため、どちらの群にも入れない。"""
    results = results_of(
        {
            "shallow": {1: 0.5, 2: 0.0, 11: 0.1, 12: 0.0},
            "deep": {1: 0.1, 2: 0.0, 11: 0.5, 12: 0.0},
            "even": {1: 0.3, 2: 0.0, 11: 0.3, 12: 0.0},
        }
    )

    selection = select_topics(results, METRIC, SHALLOW, DEEP, count=5)

    assert [choice.topic_id for choice in selection.shallow_favored] == ["shallow"]
    assert [choice.topic_id for choice in selection.deep_favored] == ["deep"]


def test_orders_each_group_by_the_size_of_the_margin_up_to_the_count() -> None:
    results = results_of(
        {
            "s1": {1: 0.1, 2: 0.0, 11: 0.0, 12: 0.0},
            "s3": {1: 0.3, 2: 0.0, 11: 0.0, 12: 0.0},
            "s2": {1: 0.2, 2: 0.0, 11: 0.0, 12: 0.0},
            "d1": {1: 0.0, 2: 0.0, 11: 0.1, 12: 0.0},
            "d3": {1: 0.0, 2: 0.0, 11: 0.3, 12: 0.0},
            "d2": {1: 0.0, 2: 0.0, 11: 0.2, 12: 0.0},
        }
    )

    selection = select_topics(results, METRIC, SHALLOW, DEEP, count=2)

    assert [choice.topic_id for choice in selection.shallow_favored] == ["s3", "s2"]
    assert [choice.topic_id for choice in selection.deep_favored] == ["d3", "d2"]


def test_rejects_a_topic_missing_a_given_layer() -> None:
    results = results_of({"0001": {1: 0.2, 2: 0.5, 11: 0.4}})

    with pytest.raises(ValueError):
        select_topics(results, METRIC, SHALLOW, DEEP, count=5)
