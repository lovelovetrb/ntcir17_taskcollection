"""画面に並べる表の中身を組み立てる。Streamlit には依存しない。"""

from __future__ import annotations

from collections.abc import Sequence

from layer_inspector.view import DocumentView, TopicView, grade_label, readable

Row = dict[str, str | int | float]


def topic_rows(topics: Sequence[TopicView], metric: str) -> list[Row]:
    return [
        {
            "トピック": topic.topic_id,
            "クエリ": readable(topic.query),
            "浅い層": topic.shallow_layer,
            f"浅い層の {metric}": topic.shallow_metrics[metric],
            "深い層": topic.deep_layer,
            f"深い層の {metric}": topic.deep_metrics[metric],
            "差": topic.margin,
        }
        for topic in topics
    ]


def metric_rows(topic: TopicView) -> list[Row]:
    """指標を行に、浅い側と深い側の層を列にする。"""
    shallow = f"層 {topic.shallow_layer}"
    deep = f"層 {topic.deep_layer}"
    return [
        {"指標": name, shallow: value, deep: topic.deep_metrics[name]}
        for name, value in topic.shallow_metrics.items()
    ]


def document_rows(
    documents: Sequence[DocumentView],
    favors_shallow: bool,
    shallow_layer: int,
    deep_layer: int,
) -> list[Row]:
    """表は検索した層の順位で並んでいるため、その順位を最初の列に置く。"""
    rows: list[Row] = []
    for document in documents:
        shallow = (f"層 {shallow_layer} の順位", document.shallow_rank)
        deep = (f"層 {deep_layer} の順位", document.deep_rank)
        first, second = (shallow, deep) if favors_shallow else (deep, shallow)
        rows.append(
            {
                first[0]: first[1],
                second[0]: second[1],
                "判定": grade_label(document.grade),
                "文書 ID": document.document_id,
            }
        )
    return rows
