"""主成分を除いた検索の記録を、本体の実験と同じ形で書き、読み戻すことを確かめる。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from analysis.pc_removal.records import (
    TopicRecord,
    VarianceRecord,
    read_topic_records,
    read_variance_records,
    summarize,
    write_results,
)
from analysis.pc_removal.run import REMOVED_COUNTS, evaluate_layer
from hidden_subspace.evaluation import align_relevance
from hidden_subspace.states import HiddenStates

MODEL_ID = "cl-tohoku/bert-base-japanese-v3"


def topic_record(topic_id: str, similarity: str, value: float) -> TopicRecord:
    return TopicRecord(
        model_id=MODEL_ID,
        layer=1,
        condition="raw",
        similarity=similarity,
        topic_id=topic_id,
        metrics={"nDCG@1000": value},
    )


RECORDS = [
    topic_record("0002", "cosine", 0.2),
    topic_record("0001", "cosine", 0.4),
    topic_record("0002", "inner_product", 0.1),
    topic_record("0001", "inner_product", 0.3),
]
VARIANCES = [VarianceRecord(model_id=MODEL_ID, layer=1, count=1, cumulative_share=0.5)]


def test_the_summary_averages_topics_for_each_similarity() -> None:
    summaries = summarize(RECORDS)

    assert [(s.similarity, s.metrics["nDCG@1000"]) for s in summaries] == [
        ("cosine", pytest.approx(0.3)),
        ("inner_product", pytest.approx(0.2)),
    ]


def test_writes_a_summary_one_file_per_topic_and_the_variances(tmp_path: Path) -> None:
    write_results(tmp_path, RECORDS, VARIANCES)

    assert len((tmp_path / "summary.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    topic_lines = (tmp_path / "topic" / "0001.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["similarity"] for line in topic_lines] == ["cosine", "inner_product"]
    assert read_variance_records(tmp_path) == VARIANCES


def test_topic_records_are_read_back_in_topic_order(tmp_path: Path) -> None:
    write_results(tmp_path, RECORDS, VARIANCES)

    assert read_topic_records(tmp_path) == [RECORDS[1], RECORDS[3], RECORDS[0], RECORDS[2]]


def test_a_missing_record_points_to_the_run_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--pooling"):
        read_topic_records(tmp_path)


def test_every_condition_is_evaluated_with_both_similarities() -> None:
    """主成分を除いたベクトルから、内積とコサインの 2 通りで検索する (ADR-0035)。"""
    torch.manual_seed(0)
    document_ids = tuple(f"d{i}" for i in range(300))
    documents = HiddenStates.of(torch.randn(300, 1, 256), item_ids=document_ids)
    queries = HiddenStates.of(torch.randn(2, 1, 256), item_ids=("0001", "0002"))
    relevance = align_relevance(
        {"0001": {"d0": 1}, "0002": {"d1": 1}}, ("0001", "0002"), document_ids
    )

    records, _ = evaluate_layer(
        MODEL_ID, documents, queries, relevance, torch.ones(300, dtype=torch.bool)
    )

    conditions = {record.condition for record in records}
    assert len(conditions) == len(REMOVED_COUNTS) + 1
    for condition in conditions:
        assert sorted(
            (record.similarity, record.topic_id)
            for record in records
            if record.condition == condition
        ) == [
            ("cosine", "0001"),
            ("cosine", "0002"),
            ("inner_product", "0001"),
            ("inner_product", "0002"),
        ]
