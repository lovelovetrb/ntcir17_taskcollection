"""TREC 形式の run ファイルを読み、層の実験と同じ評価に渡せる形にする処理を確かめる。"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from hidden_subspace.evaluation import align_relevance, evaluate
from hidden_subspace.trec_run import ranking_from_run, read_run, relevance_of_retrieved

DOCUMENTS = ("a", "b", "c", "d", "e")
TOPICS = ("0001", "0002")

LINES = "\n".join(
    [
        "0001 Q0 c 1 8.5 bm25",
        "0001 Q0 d 0 9.0 bm25",  # 行の順は順位順とは限らない
        "0002 Q0 a 0 3.0 bm25",
    ]
)


def test_reads_documents_in_rank_order(tmp_path: Path) -> None:
    path = tmp_path / "run.res"
    path.write_text(LINES, encoding="utf-8")

    run = read_run(path)

    assert run == {"0001": [("d", 9.0), ("c", 8.5)], "0002": [("a", 3.0)]}


def test_reads_gzipped_run(tmp_path: Path) -> None:
    path = tmp_path / "run.res.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(LINES)

    assert read_run(path) == {"0001": [("d", 9.0), ("c", 8.5)], "0002": [("a", 3.0)]}


def test_run_documents_come_first_and_the_rest_follow_in_collection_order() -> None:
    run = {"0001": [("d", 9.0), ("c", 8.5)]}

    ranking = ranking_from_run(run, TOPICS, DOCUMENTS)

    assert ranking.ranked_ids_for("0001") == ("d", "c", "a", "b", "e")


def test_a_topic_missing_from_the_run_keeps_collection_order() -> None:
    ranking = ranking_from_run({"0001": [("d", 9.0)]}, TOPICS, DOCUMENTS)

    assert ranking.ranked_ids_for("0002") == DOCUMENTS


def test_padded_scores_are_below_every_run_score() -> None:
    ranking = ranking_from_run({"0001": [("d", -2.0), ("c", -3.0)]}, TOPICS, DOCUMENTS)

    scores = ranking.scores[0].tolist()
    assert scores[:2] == [-2.0, -3.0]
    assert all(score < -3.0 for score in scores[2:])


def test_rejects_a_document_that_is_not_in_the_collection() -> None:
    with pytest.raises(ValueError, match="コレクションにありません"):
        ranking_from_run({"0001": [("zzz", 1.0)]}, TOPICS, DOCUMENTS)


def test_unretrieved_relevant_documents_do_not_count_as_found() -> None:
    """見つけていない適合文書は nDCG に入らないが、Recall の分母には残る。

    トピック 0001 の適合は b と d。run は d だけを 1 位で返す。b は埋めた側で 2 位に来るが
    見つけていないので、nDCG@2 は d の分だけ、Recall@2 は 2 件中 1 件になる。
    """
    run = {"0001": [("d", 9.0)]}
    ranking = ranking_from_run(run, ("0001",), DOCUMENTS)
    relevance = align_relevance({"0001": {"b": 2, "d": 1}}, ("0001",), DOCUMENTS)

    retrieved = relevance_of_retrieved(relevance, run)
    result = evaluate(ranking, retrieved, ks=(2,))

    assert retrieved.relevant_counts.tolist() == [2.0]
    assert result.scores["Recall@2"].tolist() == pytest.approx([0.5])
    assert result.scores["nDCG@2"].tolist() == pytest.approx(
        [1.0 / (1.0 + 1.0 / 1.584962500721156)]
    )


def test_relevance_of_retrieved_leaves_the_original_untouched() -> None:
    run = {"0001": [("d", 9.0)]}
    relevance = align_relevance({"0001": {"b": 2, "d": 1}}, ("0001",), DOCUMENTS)

    relevance_of_retrieved(relevance, run)

    assert relevance.values.sum().item() == 2.0
