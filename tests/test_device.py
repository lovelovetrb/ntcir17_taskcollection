"""GPU 上でも一連の処理が動くことを確かめる。

合成データのテストは CPU で回るため、置き場所の食い違いはここでしか出ない。
GPU がなければ飛ばす。
"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace.evaluation import align_relevance, evaluate
from hidden_subspace.retrieval import rank_documents
from hidden_subspace.states import HiddenStates
from hidden_subspace.vectors import build_search_vectors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU がない")

DOCUMENTS = ("a", "b", "c", "d")
TOPICS = ("0001", "0002")


def test_selection_works_on_a_device() -> None:
    """絞り込みで作る添字は、値と同じ置き場所になければならない。"""
    states = HiddenStates.of(torch.randn(4, 5, 8), item_ids=DOCUMENTS).to("cuda")

    narrowed = states.select_layers([1, 3]).select_dimensions([0, 2])

    assert narrowed.values.shape == (4, 2, 2)


def test_the_whole_path_works_on_a_device() -> None:
    """隠れ状態から指標まで、置き場所を揃えたまま通る。"""
    documents = HiddenStates.of(torch.randn(4, 5, 8), item_ids=DOCUMENTS).to("cuda")
    queries = HiddenStates.of(torch.randn(2, 5, 8), item_ids=TOPICS).to("cuda")
    relevance = align_relevance({"0001": {"a": 2}, "0002": {"c": 1}}, TOPICS, DOCUMENTS).to("cuda")

    narrowed_documents = documents.select_layers([1]).select_dimensions(range(4))
    narrowed_queries = queries.select_layers([1]).select_dimensions(range(4))
    ranking = rank_documents(
        build_search_vectors(narrowed_queries, normalize_layers=False),
        build_search_vectors(narrowed_documents, normalize_layers=False),
    )
    result = evaluate(ranking, relevance, ks=(1, 4))

    assert set(result.macro_average()) == {
        "nDCG@1",
        "nDCG@4",
        "Precision@1",
        "Precision@4",
        "Recall@1",
        "Recall@4",
    }
