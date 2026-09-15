"""画面に並べる表の中身を組み立てる処理を確かめる。画面そのものは確かめない。"""

from __future__ import annotations

import pytest

from layer_inspector.documents import RankedDocument
from layer_inspector.presentation import document_rows, grade_label


@pytest.mark.parametrize(
    ("grade", "label"),
    [(2, "A"), (1, "B"), (0, "C"), (None, "未判定")],
)
def test_grades_are_shown_with_the_labels_of_the_distributed_judgements(
    grade: int | None, label: str
) -> None:
    """配布データは A を適合、B を部分的に適合、C を非適合と表記している。"""
    assert grade_label(grade) == label


def test_document_rows_lead_with_the_rank_on_the_layer_that_was_searched() -> None:
    """表は検索した層の順位で並んでおり、その順位が最初の列にないと並び順が読めない。"""
    documents = [RankedDocument("d1", shallow_rank=3, deep_rank=120, grade=None)]

    shallow = document_rows(documents, favors_shallow=True, shallow_layer=1, deep_layer=12)
    deep = document_rows(documents, favors_shallow=False, shallow_layer=1, deep_layer=12)

    assert shallow == [{"層 1 の順位": 3, "層 12 の順位": 120, "判定": "未判定", "文書 ID": "d1"}]
    assert list(shallow[0]) == ["層 1 の順位", "層 12 の順位", "判定", "文書 ID"]
    assert list(deep[0]) == ["層 12 の順位", "層 1 の順位", "判定", "文書 ID"]
