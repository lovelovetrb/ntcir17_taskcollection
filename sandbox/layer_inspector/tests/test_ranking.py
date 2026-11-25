"""選んだトピックが使う層で検索し直す処理を確かめる。"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace.experiment.results import Configuration, DimensionChoice, TopicResult
from hidden_subspace.states import HiddenStates
from layer_inspector.ranking import rank_selected_layers
from layer_inspector.selection import TopicChoice, TopicSelection

# 文書 a は層 0〜2 でクエリに近く、文書 b は層 3 でだけクエリに近い。
DOCUMENTS = HiddenStates.of(
    torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    ),
    item_ids=("a", "b"),
)
QUERIES = HiddenStates.of(torch.tensor([[[1.0, 0.0]] * 4] * 2), item_ids=("0001", "0002"))


def result_of(topic_id: str, layer: int, dimensions: DimensionChoice | None = None) -> TopicResult:
    configuration = Configuration(
        model_id="cl-tohoku/bert-base-japanese-v3",
        layers=(layer,),
        dimensions=dimensions or DimensionChoice(kind="all"),
        normalize_layers=False,
        similarity="cosine",
    )
    return TopicResult("layer-sweep", configuration, topic_id, {"nDCG@1000": 0.0})


def choice_of(topic_id: str, shallow_layer: int, deep_layer: int) -> TopicChoice:
    return TopicChoice(
        topic_id=topic_id,
        shallow=result_of(topic_id, shallow_layer),
        deep=result_of(topic_id, deep_layer),
        margin=0.1,
    )


def test_ranks_each_layer_used_by_the_chosen_topics_and_no_other() -> None:
    selection = TopicSelection(
        shallow_favored=[choice_of("0001", 1, 3)],
        deep_favored=[choice_of("0002", 1, 2)],
    )

    rankings = rank_selected_layers(selection, DOCUMENTS, QUERIES)

    assert sorted(rankings) == [1, 2, 3]


def test_each_ranking_is_made_from_the_hidden_states_of_its_layer() -> None:
    """キーの層と中身の層が食い違うと、別の層の検索結果を読むことになる。"""
    selection = TopicSelection(shallow_favored=[choice_of("0001", 1, 3)], deep_favored=[])

    rankings = rank_selected_layers(selection, DOCUMENTS, QUERIES)

    assert rankings[1].ranked_ids_for("0001")[0] == "a"
    assert rankings[3].ranked_ids_for("0001")[0] == "b"


def test_rejects_records_whose_dimensions_were_narrowed() -> None:
    """このビューアは層ごとの分析だけを扱い、次元を絞った構成は検索し直さない。"""
    narrowed = result_of("0001", 1, DimensionChoice(kind="random", parameters={"count": 1}))
    choice = TopicChoice("0001", shallow=narrowed, deep=result_of("0001", 3), margin=0.1)
    selection = TopicSelection(shallow_favored=[choice], deep_favored=[])

    with pytest.raises(ValueError):
        rank_selected_layers(selection, DOCUMENTS, QUERIES)


def test_a_standardized_record_is_ranked_again_with_standardization() -> None:
    """記録と同じ条件で検索し直す。

    生のコサインでは b がクエリに最も近く、文書の平均と標準偏差で直すと a が最も近くなる。
    """
    documents = HiddenStates.of(
        torch.tensor([[[12.0, 0.0]], [[8.0, 1.0]], [[10.0, -1.0]]]), item_ids=("a", "b", "c")
    )
    queries = HiddenStates.of(torch.tensor([[[11.0, 1.0]]]), item_ids=("0001",))
    configuration = Configuration(
        model_id="cl-tohoku/bert-base-japanese-v3",
        layers=(0,),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
        similarity="cosine",
        standardize=True,
    )
    result = TopicResult("layer-sweep-standardized", configuration, "0001", {"nDCG@1000": 0.0})
    choice = TopicChoice("0001", shallow=result, deep=result, margin=0.1)
    selection = TopicSelection(shallow_favored=[choice], deep_favored=[])

    rankings = rank_selected_layers(selection, documents, queries)

    assert rankings[0].ranked_ids_for("0001")[0] == "a"
