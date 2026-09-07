"""順位を指標に変換する処理を確かめる。"""

from __future__ import annotations

import math

import pytest
import torch

from hidden_subspace.evaluation import align_relevance, evaluate
from hidden_subspace.retrieval import Ranking

DOCUMENTS = ("a", "b", "c", "d", "e")
TOPICS = ("t1",)


def ranking_of(order: list[list[int]], topics: tuple[str, ...] = TOPICS) -> Ranking:
    return Ranking(
        topic_ids=topics,
        document_ids=DOCUMENTS,
        order=torch.tensor(order),
        scores=torch.zeros(len(topics), len(DOCUMENTS)),
    )


def test_alignment_marks_relevant_documents_in_document_order() -> None:
    relevance = align_relevance({"t1": {"b": 2, "d": 1, "a": 0}}, TOPICS, DOCUMENTS)

    assert relevance.values[0].tolist() == [0, 1, 0, 1, 0]


def test_alignment_treats_unjudged_documents_as_not_relevant() -> None:
    """判定のない文書は非適合として数える (ADR-0007)。"""
    relevance = align_relevance({"t1": {"b": 2}}, TOPICS, DOCUMENTS)

    assert relevance.values[0].tolist() == [0, 1, 0, 0, 0]


def test_relevant_count_includes_documents_outside_the_corpus() -> None:
    """検索できない適合文書も分母に入れる。qrels の通りに数えるため。"""
    relevance = align_relevance({"t1": {"b": 2, "not-in-corpus": 2}}, TOPICS, DOCUMENTS)

    assert relevance.relevant_counts[0].item() == 2
    assert relevance.values[0].sum().item() == 1


def test_precision_counts_relevant_documents_in_the_top_k() -> None:
    relevance = align_relevance({"t1": {"b": 1, "d": 1}}, TOPICS, DOCUMENTS)
    ranking = ranking_of([[1, 0, 3, 2, 4]])  # b, a, d, c, e

    result = evaluate(ranking, relevance, ks=(1, 2, 4))

    assert result.macro_average()["Precision@1"] == pytest.approx(1.0)
    assert result.macro_average()["Precision@2"] == pytest.approx(0.5)
    assert result.macro_average()["Precision@4"] == pytest.approx(0.5)


def test_recall_divides_by_the_number_of_relevant_documents() -> None:
    relevance = align_relevance({"t1": {"b": 1, "d": 1}}, TOPICS, DOCUMENTS)
    ranking = ranking_of([[1, 0, 3, 2, 4]])  # b, a, d, c, e

    result = evaluate(ranking, relevance, ks=(1, 4))

    assert result.macro_average()["Recall@1"] == pytest.approx(0.5)
    assert result.macro_average()["Recall@4"] == pytest.approx(1.0)


def test_ndcg_discounts_lower_positions() -> None:
    """手計算と突き合わせる。順位 i の利得を log2(i+1) で割って足す。"""
    relevance = align_relevance({"t1": {"a": 1, "c": 1}}, TOPICS, DOCUMENTS)
    ranking = ranking_of([[0, 1, 2, 3, 4]])  # a, b, c, d, e → 適合は 1 位と 3 位

    result = evaluate(ranking, relevance, ks=(3,))

    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert result.macro_average()["nDCG@3"] == pytest.approx(dcg / idcg)


def test_ndcg_reaches_one_when_relevant_documents_come_first() -> None:
    relevance = align_relevance({"t1": {"a": 1, "b": 1}}, TOPICS, DOCUMENTS)
    ranking = ranking_of([[0, 1, 2, 3, 4]])

    result = evaluate(ranking, relevance, ks=(2, 5))

    assert result.macro_average()["nDCG@2"] == pytest.approx(1.0)
    assert result.macro_average()["nDCG@5"] == pytest.approx(1.0)


def test_k_larger_than_the_number_of_documents_uses_every_document() -> None:
    relevance = align_relevance({"t1": {"a": 1}}, TOPICS, DOCUMENTS)
    ranking = ranking_of([[0, 1, 2, 3, 4]])

    result = evaluate(ranking, relevance, ks=(999,))

    assert result.macro_average()["Recall@999"] == pytest.approx(1.0)
    assert result.macro_average()["Precision@999"] == pytest.approx(1 / 5)


def test_a_topic_without_relevant_documents_scores_zero() -> None:
    """0 で割らずに 0 とする。NTCIR-1 には該当トピックがないが、静かに壊れないように。"""
    relevance = align_relevance({"t1": {"a": 0}}, TOPICS, DOCUMENTS)
    ranking = ranking_of([[0, 1, 2, 3, 4]])

    result = evaluate(ranking, relevance, ks=(3,))

    assert result.macro_average()["Recall@3"] == pytest.approx(0.0)
    assert result.macro_average()["nDCG@3"] == pytest.approx(0.0)


def test_macro_average_takes_the_mean_over_topics() -> None:
    """トピックごとに指標を出してから平均する (ADR-0013)。"""
    relevance = align_relevance({"t1": {"a": 1}, "t2": {"e": 1}}, ("t1", "t2"), DOCUMENTS)
    ranking = ranking_of([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], topics=("t1", "t2"))

    result = evaluate(ranking, relevance, ks=(1,))

    assert result.scores["Precision@1"].tolist() == [1.0, 0.0]
    assert result.macro_average()["Precision@1"] == pytest.approx(0.5)


def test_rejects_a_ranking_built_for_other_documents() -> None:
    relevance = align_relevance({"t1": {"a": 1}}, TOPICS, ("x", "y"))
    ranking = ranking_of([[0, 1, 2, 3, 4]])

    with pytest.raises(ValueError):
        evaluate(ranking, relevance, ks=(1,))


def test_relevance_can_be_moved_to_a_device() -> None:
    relevance = align_relevance({"t1": {"b": 1}}, TOPICS, DOCUMENTS)

    moved = relevance.to("cpu")

    assert moved.topic_ids == relevance.topic_ids
    assert torch.equal(moved.values, relevance.values)
    assert torch.equal(moved.relevant_counts, relevance.relevant_counts)
