"""mMARCO の三つ組から Tevatron の学習データを作る処理を確かめる。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpr.data import (
    check_ids,
    group_triples,
    prepare_training_data,
    read_texts,
    training_rows,
    write_rows,
)

TRIPLES = [
    "q1\tp1\tn1\n",
    "q1\tp1\tn2\n",
    "q2\tp3\tn1\n",
    "q1\tp2\tn1\n",
]
QUERIES = {"q1": "質問 1", "q2": "質問 2"}
PASSAGES = {"p1": "答え 1", "p2": "答え 2", "p3": "答え 3", "n1": "外れ 1", "n2": "外れ 2"}


def test_triples_are_grouped_by_query_without_duplicates() -> None:
    """負例は組になったものすべてを重複なしで、適合文書が複数あればすべてを入れる。"""
    groups = group_triples(TRIPLES)

    assert list(groups) == ["q1", "q2"]
    assert groups["q1"].positives == ["p1", "p2"]
    assert groups["q1"].negatives == ["n1", "n2"]
    assert groups["q2"].negatives == ["n1"]


def test_texts_keep_tabs_after_the_identifier() -> None:
    assert read_texts(["p1\t前\t後\n"]) == {"p1": "前\t後"}


def test_ids_missing_from_the_texts_are_rejected() -> None:
    """欠けたまま進めると、学習データから文が黙って抜ける。"""
    groups = group_triples(TRIPLES)

    with pytest.raises(ValueError, match="文書集合 \\(1 件\\)"):
        check_ids(groups, QUERIES, {k: v for k, v in PASSAGES.items() if k != "n2"})
    with pytest.raises(ValueError, match="クエリ \\(1 件\\)"):
        check_ids(groups, {"q1": "質問 1"}, PASSAGES)
    check_ids(groups, QUERIES, PASSAGES)


def test_a_row_is_in_the_format_tevatron_reads() -> None:
    rows = list(training_rows(group_triples(TRIPLES), QUERIES, PASSAGES))

    assert rows[0] == {
        "query_id": "q1",
        "query": "質問 1",
        "positive_passages": [
            {"docid": "p1", "text": "答え 1"},
            {"docid": "p2", "text": "答え 2"},
        ],
        "negative_passages": [
            {"docid": "n1", "text": "外れ 1"},
            {"docid": "n2", "text": "外れ 2"},
        ],
    }


def test_rows_are_written_one_per_line_without_a_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"

    count = write_rows(training_rows(group_triples(TRIPLES), QUERIES, PASSAGES), path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert count == len(lines) == 2
    assert json.loads(lines[1])["query"] == "質問 2"
    assert list(tmp_path.iterdir()) == [path]


def test_existing_training_data_is_used_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("既存\n", encoding="utf-8")

    assert prepare_training_data(tmp_path / "absent", path) == path
    assert path.read_text(encoding="utf-8") == "既存\n"


def test_missing_downloads_point_to_the_download_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"dpr\.download"):
        prepare_training_data(tmp_path, tmp_path / "train.jsonl")
