"""画面に並べる表の中身を組み立てる処理を確かめる。画面そのものは確かめない。"""

from __future__ import annotations

from layer_inspector.presentation import document_rows, topic_rows
from layer_inspector.view import DocumentView, TopicView


def topic_of() -> TopicView:
    return TopicView(
        topic_id="0001",
        favors_shallow=True,
        query="[CLS] メディア 同期 [SEP]",
        shallow_layer=1,
        deep_layer=12,
        shallow_metrics={"nDCG@1000": 0.5},
        deep_metrics={"nDCG@1000": 0.1},
        margin=0.4,
        top=[],
        relevant_top=[],
        relevant_bottom=[],
    )


def test_document_rows_lead_with_the_rank_on_the_layer_that_was_searched() -> None:
    """表は検索した層の順位で並んでおり、その順位が最初の列にないと並び順が読めない。"""
    documents = [DocumentView("d1", shallow_rank=3, deep_rank=120, grade=None)]

    shallow = document_rows(documents, favors_shallow=True, shallow_layer=1, deep_layer=12)
    deep = document_rows(documents, favors_shallow=False, shallow_layer=1, deep_layer=12)

    assert shallow == [{"層 1 の順位": 3, "層 12 の順位": 120, "判定": "未判定", "文書 ID": "d1"}]
    assert list(shallow[0]) == ["層 1 の順位", "層 12 の順位", "判定", "文書 ID"]
    assert list(deep[0]) == ["層 12 の順位", "層 1 の順位", "判定", "文書 ID"]


def test_the_query_column_is_readable() -> None:
    """decode した文字列はトークンの区切りに空白が入り、そのままでは文として読めない。"""
    rows = topic_rows([topic_of()], "nDCG@1000")

    assert rows[0]["クエリ"] == "[CLS]メディア同期[SEP]"
