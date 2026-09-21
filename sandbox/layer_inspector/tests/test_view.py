"""結果ファイルを表示用に読む処理を確かめる。

表示側は実験の型に戻さず、表を描くのに要る値だけを読む (ADR-0021)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from layer_inspector.view import grade_label, read_view, readable

MODEL_ID = "cl-tohoku/bert-base-japanese-v3"


def topic_result(topic_id: str, layer: int, score: float) -> dict[str, Any]:
    """作成側が書き出す、実験の記録と同じ形のトピックの結果。"""
    return {
        "experiment": "layer-sweep",
        "model_id": MODEL_ID,
        "layers": [layer],
        "dimensions": {"kind": "all"},
        "normalize_layers": False,
        "topic_id": topic_id,
        "metrics": {"nDCG@10": score / 2, "nDCG@1000": score},
    }


def document(document_id: str, shallow: int, deep: int, grade: int | None) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "shallow_rank": shallow,
        "deep_rank": deep,
        "grade": grade,
    }


def write_result_file(path: Path) -> None:
    record = {
        "settings": {
            "model_id": MODEL_ID,
            "experiment": "layer-sweep",
            "metric": "nDCG@1000",
            "shallow_layers": [1, 2],
            "deep_layers": [11, 12],
            "topic_count": 5,
            "top_count": 10,
            "relevant_count": 5,
        },
        "shallow_favored": [
            {
                "topic_id": "0001",
                "shallow": topic_result("0001", 1, 0.5),
                "deep": topic_result("0001", 12, 0.1),
                "margin": 0.4,
            }
        ],
        "deep_favored": [
            {
                "topic_id": "0002",
                "shallow": topic_result("0002", 2, 0.1),
                "deep": topic_result("0002", 11, 0.3),
                "margin": -0.2,
            }
        ],
        "documents": {
            "0001": {
                "topic_id": "0001",
                "top": [document("d1", 1, 40, None)],
                "relevant_top": [document("d2", 3, 2, 2)],
                "relevant_bottom": [],
            },
            "0002": {
                "topic_id": "0002",
                "top": [document("d3", 90, 1, 1)],
                "relevant_top": [],
                "relevant_bottom": [],
            },
        },
        "inputs": {
            "queries": {"0001": "[CLS] メディア 同期 [SEP]", "0002": "[CLS] 輻輳 制御 [SEP]"},
            "documents": {"d1": "[CLS] 題名 抄録 [SEP]", "d2": "[CLS] 別 の 文書 [SEP]"},
        },
    }
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_reads_the_topics_of_both_groups_with_their_layers_and_metrics(tmp_path: Path) -> None:
    path = tmp_path / "inspection.json"
    write_result_file(path)

    view = read_view(path)

    shallow = view.shallow_favored[0]
    deep = view.deep_favored[0]
    assert view.metric == "nDCG@1000"
    assert (shallow.topic_id, shallow.favors_shallow) == ("0001", True)
    assert (shallow.shallow_layer, shallow.deep_layer) == (1, 12)
    assert shallow.shallow_metrics["nDCG@1000"] == pytest.approx(0.5)
    assert shallow.margin == pytest.approx(0.4)
    assert shallow.query == "[CLS] メディア 同期 [SEP]"
    assert (deep.topic_id, deep.favors_shallow) == ("0002", False)
    assert (deep.shallow_layer, deep.deep_layer) == (2, 11)


def test_reads_the_documents_with_both_ranks_and_the_grade(tmp_path: Path) -> None:
    """判定を受けていない文書は grade がなく、表では「未判定」と出す。"""
    path = tmp_path / "inspection.json"
    write_result_file(path)

    view = read_view(path)

    topic = view.shallow_favored[0]
    assert [d.document_id for d in topic.top] == ["d1"]
    assert (topic.top[0].shallow_rank, topic.top[0].deep_rank, topic.top[0].grade) == (1, 40, None)
    assert topic.relevant_top[0].grade == 2
    assert topic.relevant_bottom == []
    assert view.documents["d1"] == "[CLS] 題名 抄録 [SEP]"


def test_a_missing_result_file_names_the_command_that_builds_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"layer_inspector\.build"):
        read_view(tmp_path / "absent.json")


@pytest.mark.parametrize(
    ("grade", "label"),
    [(2, "A"), (1, "B"), (0, "C"), (None, "未判定")],
)
def test_grades_are_shown_with_the_labels_of_the_distributed_judgements(
    grade: int | None, label: str
) -> None:
    """配布データは A を適合、B を部分的に適合、C を非適合と表記している。"""
    assert grade_label(grade) == label


def test_the_spaces_between_tokens_are_removed_for_reading() -> None:
    """日本語のトークナイザは decode でトークンの区切りに空白を入れ、文として読めない。"""
    assert readable("[CLS] メディア 同期 [SEP]") == "[CLS]メディア同期[SEP]"
