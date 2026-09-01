"""検索ベクトルから順位を付ける処理を確かめる。"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace.retrieval import rank_documents
from hidden_subspace.vectors import SearchVectors


def vectors_of(
    values: torch.Tensor,
    ids: tuple[str, ...],
    *,
    layers: tuple[int, ...] = (0,),
    dimensions: tuple[int, ...] = (0, 1),
) -> SearchVectors:
    return SearchVectors(
        values=values,
        item_ids=ids,
        layer_indices=layers,
        dimension_indices=dimensions,
        zero_length_count=0,
    )


def test_orders_documents_by_descending_score() -> None:
    queries = vectors_of(torch.tensor([[1.0, 0.0]]), ("q",))
    documents = vectors_of(
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]]), ("far", "near", "middle")
    )

    ranking = rank_documents(queries, documents)

    assert ranking.ranked_ids_for("q") == ("near", "middle", "far")


def test_scores_are_the_inner_product() -> None:
    queries = vectors_of(torch.tensor([[1.0, 0.0]]), ("q",))
    documents = vectors_of(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ("a", "b"))

    ranking = rank_documents(queries, documents)

    torch.testing.assert_close(ranking.scores[0], torch.tensor([1.0, 0.0]))


def test_ties_are_broken_by_document_order() -> None:
    """同点の順序が実行ごとに揺れると再現できない。文書の並び順で決める。"""
    queries = vectors_of(torch.tensor([[0.0, 0.0]]), ("q",))
    documents = vectors_of(torch.zeros(4, 2), ("a", "b", "c", "d"))

    ranking = rank_documents(queries, documents)

    assert ranking.ranked_ids_for("q") == ("a", "b", "c", "d")


def test_every_document_appears_exactly_once() -> None:
    """順位は打ち切らない。上位何件を見るかは指標の側が決める。"""
    queries = vectors_of(torch.randn(2, 2), ("q1", "q2"))
    documents = vectors_of(torch.randn(5, 2), ("a", "b", "c", "d", "e"))

    ranking = rank_documents(queries, documents)

    for topic in ("q1", "q2"):
        assert sorted(ranking.ranked_ids_for(topic)) == ["a", "b", "c", "d", "e"]


def test_rejects_vectors_built_from_different_selections() -> None:
    """クエリと文書の構成が食い違うと、内積は無意味になるのに例外は出ない。"""
    queries = vectors_of(torch.randn(1, 2), ("q",), layers=(0,), dimensions=(0, 1))
    documents = vectors_of(torch.randn(3, 2), ("a", "b", "c"), layers=(2,), dimensions=(0, 1))

    with pytest.raises(ValueError):
        rank_documents(queries, documents)


def test_rejects_mismatched_dimensionality() -> None:
    queries = vectors_of(torch.randn(1, 2), ("q",), dimensions=(0, 1))
    documents = vectors_of(torch.randn(3, 3), ("a", "b", "c"), dimensions=(0, 1, 2))

    with pytest.raises(ValueError):
        rank_documents(queries, documents)


def test_carries_over_the_identifiers() -> None:
    queries = vectors_of(torch.randn(2, 2), ("q1", "q2"))
    documents = vectors_of(torch.randn(3, 2), ("a", "b", "c"))

    ranking = rank_documents(queries, documents)

    assert ranking.topic_ids == ("q1", "q2")
    assert ranking.document_ids == ("a", "b", "c")
