"""検索し直した順位を、layer-sweep の記録と照合する処理を確かめる。"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from hidden_subspace.evaluation import align_relevance
from hidden_subspace.experiment.results import Configuration, DimensionChoice, TopicResult
from hidden_subspace.retrieval import Ranking
from layer_inspector.selection import TopicChoice, TopicSelection
from layer_inspector.verification import verify_against_records

TOPICS = ("0001",)
DOCUMENTS = ("a", "b")

# 文書は 2 件で、適合するのは 1 位の a だけ。k が文書数を超えると、上位 2 件で数える。
RELEVANT_FIRST = {
    "Precision@1": 1.0,
    "Recall@1": 1.0,
    "nDCG@1": 1.0,
    "Precision@10": 0.5,
    "Recall@10": 1.0,
    "nDCG@10": 1.0,
    "Precision@1000": 0.5,
    "Recall@1000": 1.0,
    "nDCG@1000": 1.0,
}
RELEVANCE = align_relevance({"0001": {"a": 2}}, TOPICS, DOCUMENTS)
RANKING = Ranking(
    topic_ids=TOPICS,
    document_ids=DOCUMENTS,
    order=torch.tensor([[0, 1]]),
    scores=torch.zeros(1, 2),
)


def result_of(layer: int, metrics: Mapping[str, float]) -> TopicResult:
    configuration = Configuration(
        model_id="cl-tohoku/bert-base-japanese-v3",
        layers=(layer,),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
    )
    return TopicResult("layer-sweep", configuration, "0001", metrics)


def selection_of(shallow: Mapping[str, float], deep: Mapping[str, float]) -> TopicSelection:
    choice = TopicChoice(
        topic_id="0001", shallow=result_of(1, shallow), deep=result_of(11, deep), margin=0.1
    )
    return TopicSelection(shallow_favored=[choice], deep_favored=[])


def test_accepts_rankings_that_reproduce_the_records() -> None:
    selection = selection_of(RELEVANT_FIRST, RELEVANT_FIRST)

    verify_against_records(selection, {1: RANKING, 11: RANKING}, RELEVANCE)


def test_names_the_topic_and_metric_that_do_not_match() -> None:
    recorded = {**RELEVANT_FIRST, "nDCG@10": 0.9}
    selection = selection_of(recorded, RELEVANT_FIRST)

    with pytest.raises(ValueError) as raised:
        verify_against_records(selection, {1: RANKING, 11: RANKING}, RELEVANCE)

    assert "0001" in str(raised.value)
    assert "nDCG@10" in str(raised.value)


def test_checks_the_deep_side_as_well() -> None:
    """浅い側だけを見て終えると、深い層の順位が変わっていても気づけない。"""
    selection = selection_of(RELEVANT_FIRST, {**RELEVANT_FIRST, "Recall@1000": 0.0})

    with pytest.raises(ValueError):
        verify_against_records(selection, {1: RANKING, 11: RANKING}, RELEVANCE)
