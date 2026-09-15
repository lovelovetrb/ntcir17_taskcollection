"""検索結果の上位と適合文書を、両方の層での順位つきで取り出す処理を確かめる。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from hidden_subspace.experiment.results import Configuration, DimensionChoice, TopicResult
from hidden_subspace.retrieval import Ranking
from layer_inspector.documents import RankedDocument, extract_documents
from layer_inspector.selection import TopicChoice, TopicSelection

SHALLOW_LAYER = 1
DEEP_LAYER = 11


def ranking_of(document_ids: Sequence[str], order: Sequence[int]) -> Ranking:
    return Ranking(
        topic_ids=("0001",),
        document_ids=tuple(document_ids),
        order=torch.tensor([list(order)]),
        scores=torch.zeros(1, len(document_ids)),
    )


def result_of(layer: int) -> TopicResult:
    configuration = Configuration(
        model_id="cl-tohoku/bert-base-japanese-v3",
        layers=(layer,),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
    )
    return TopicResult("layer-sweep", configuration, "0001", {"nDCG@1000": 0.0})


def selection_of(margin: float) -> TopicSelection:
    """差が正なら浅い層が優位、負なら深い層が優位な群に入れる。"""
    choice = TopicChoice(
        topic_id="0001",
        shallow=result_of(SHALLOW_LAYER),
        deep=result_of(DEEP_LAYER),
        margin=margin,
    )
    if margin > 0:
        return TopicSelection(shallow_favored=[choice], deep_favored=[])
    return TopicSelection(shallow_favored=[], deep_favored=[choice])


def ids_of(documents: Sequence[RankedDocument]) -> list[str]:
    return [document.document_id for document in documents]


FIVE = ("d0", "d1", "d2", "d3", "d4")
# 浅い層では d0 から、深い層では d4 から並ぶ。
OPPOSITE_RANKINGS: Mapping[int, Ranking] = {
    SHALLOW_LAYER: ranking_of(FIVE, [0, 1, 2, 3, 4]),
    DEEP_LAYER: ranking_of(FIVE, [4, 3, 2, 1, 0]),
}


def test_top_documents_follow_the_layer_that_is_favored() -> None:
    shallow = extract_documents(selection_of(0.1), OPPOSITE_RANKINGS, {}, 3, 5)
    deep = extract_documents(selection_of(-0.1), OPPOSITE_RANKINGS, {}, 3, 5)

    assert ids_of(shallow["0001"].top) == ["d0", "d1", "d2"]
    assert ids_of(deep["0001"].top) == ["d4", "d3", "d2"]


def test_each_document_carries_its_rank_on_both_layers_and_its_grade() -> None:
    qrels = {"0001": {"d0": 2, "d1": 0}}

    documents = extract_documents(selection_of(0.1), OPPOSITE_RANKINGS, qrels, 3, 5)

    assert documents["0001"].top == [
        RankedDocument("d0", shallow_rank=1, deep_rank=5, grade=2),
        RankedDocument("d1", shallow_rank=2, deep_rank=4, grade=0),
        RankedDocument("d2", shallow_rank=3, deep_rank=3, grade=None),
    ]


TWENTY = tuple(f"d{i}" for i in range(20))
# 浅い層では番号の大きい文書ほど上位に来る。
REVERSED_RANKINGS: Mapping[int, Ranking] = {
    SHALLOW_LAYER: ranking_of(TWENTY, list(reversed(range(20)))),
    DEEP_LAYER: ranking_of(TWENTY, list(range(20))),
}


@pytest.mark.parametrize(
    ("relevant", "expected_top", "expected_bottom"),
    [
        (12, ["d11", "d10", "d9", "d8", "d7"], ["d4", "d3", "d2", "d1", "d0"]),
        (7, ["d6", "d5", "d4", "d3", "d2"], ["d1", "d0"]),
        (3, ["d2", "d1", "d0"], []),
    ],
)
def test_relevant_documents_are_split_without_overlap(
    relevant: int, expected_top: list[str], expected_bottom: list[str]
) -> None:
    """適合文書が少ないと上位と下位に同じ文書が出て、下位が取り落とした文書を表さなくなる。"""
    qrels = {"0001": {f"d{i}": 1 for i in range(relevant)}}

    documents = extract_documents(selection_of(0.1), REVERSED_RANKINGS, qrels, 10, 5)

    assert ids_of(documents["0001"].relevant_top) == expected_top
    assert ids_of(documents["0001"].relevant_bottom) == expected_bottom


def test_documents_judged_not_relevant_are_not_counted_as_relevant() -> None:
    qrels = {"0001": {"d0": 2, "d1": 1, "d2": 0}}

    documents = extract_documents(selection_of(0.1), OPPOSITE_RANKINGS, qrels, 3, 5)

    assert ids_of(documents["0001"].relevant_top) == ["d0", "d1"]
