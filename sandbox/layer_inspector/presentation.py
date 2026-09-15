"""画面に並べる表の中身を組み立てる。Streamlit には依存しない。"""

from __future__ import annotations

from collections.abc import Sequence

from hidden_subspace.corpus.judgements import GRADE_VALUES
from hidden_subspace.experiment.results import TopicResult
from layer_inspector.documents import RankedDocument
from layer_inspector.inputs import ModelInputs
from layer_inspector.selection import TopicChoice

GRADE_LABELS = {value: label for label, value in GRADE_VALUES.items()}
"""grade から配布データの表記 (A / B / C) への対応。"""
UNJUDGED = "未判定"

Row = dict[str, str | int | float]


def grade_label(grade: int | None) -> str:
    return UNJUDGED if grade is None else GRADE_LABELS[grade]


def layer_of(result: TopicResult) -> int:
    (layer,) = result.configuration.layers
    return layer


def topic_rows(choices: Sequence[TopicChoice], inputs: ModelInputs, metric: str) -> list[Row]:
    return [
        {
            "トピック": choice.topic_id,
            "クエリ": inputs.queries[choice.topic_id],
            "浅い層": layer_of(choice.shallow),
            f"浅い層の {metric}": choice.shallow.metrics[metric],
            "深い層": layer_of(choice.deep),
            f"深い層の {metric}": choice.deep.metrics[metric],
            "差": choice.margin,
        }
        for choice in choices
    ]


def metric_rows(choice: TopicChoice) -> list[Row]:
    """指標を行に、浅い側と深い側の層を列にする。"""
    shallow = f"層 {layer_of(choice.shallow)}"
    deep = f"層 {layer_of(choice.deep)}"
    return [
        {"指標": name, shallow: value, deep: choice.deep.metrics[name]}
        for name, value in choice.shallow.metrics.items()
    ]


def document_rows(
    documents: Sequence[RankedDocument],
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
