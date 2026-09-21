"""結果ファイルを読み、画面に並べるための型に移す。

表示側は `hidden_subspace`・torch・transformers のいずれも import しない (ADR-0021)。
作成側は実験の記録と同じ形で書き出すが、ここでは表を描くのに要る値だけを読む。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUILD_COMMAND = "python -m layer_inspector.build"
GRADE_LABELS = {2: "A", 1: "B", 0: "C"}
UNJUDGED = "未判定"


@dataclass(frozen=True)
class DocumentView:
    document_id: str
    shallow_rank: int
    """浅い側の層での順位。1 始まり。"""
    deep_rank: int
    """深い側の層での順位。1 始まり。"""
    grade: int | None
    """qrels の grade。判定を受けていない文書は None。"""


@dataclass(frozen=True)
class TopicView:
    topic_id: str
    favors_shallow: bool
    """浅い層が優位なら True。上位と適合文書は、優位な側の層の順位で並んでいる。"""
    query: str
    shallow_layer: int
    deep_layer: int
    shallow_metrics: dict[str, float]
    deep_metrics: dict[str, float]
    margin: float
    top: list[DocumentView]
    relevant_top: list[DocumentView]
    relevant_bottom: list[DocumentView]

    @property
    def searched_layer(self) -> int:
        return self.shallow_layer if self.favors_shallow else self.deep_layer

    @property
    def other_layer(self) -> int:
        return self.deep_layer if self.favors_shallow else self.shallow_layer


@dataclass(frozen=True)
class InspectionView:
    model_id: str
    metric: str
    shallow_favored: list[TopicView]
    deep_favored: list[TopicView]
    documents: dict[str, str]
    """文書 ID から、モデルに入力された文書への対応。"""


def read_view(path: Path) -> InspectionView:
    if not path.is_file():
        raise FileNotFoundError(
            f"結果ファイルがありません: {path}\n先に {BUILD_COMMAND} で作ってください。"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    settings = record["settings"]
    queries = record["inputs"]["queries"]
    documents = record["documents"]
    return InspectionView(
        model_id=settings["model_id"],
        metric=settings["metric"],
        shallow_favored=[
            _topic(choice, True, queries, documents) for choice in record["shallow_favored"]
        ],
        deep_favored=[
            _topic(choice, False, queries, documents) for choice in record["deep_favored"]
        ],
        documents=dict(record["inputs"]["documents"]),
    )


def grade_label(grade: int | None) -> str:
    return UNJUDGED if grade is None else GRADE_LABELS[grade]


def readable(text: str) -> str:
    """トークンの区切りの空白を取り除く。日本語では、空白のまま並べると文として読めない。"""
    return text.replace(" ", "")


def _topic(
    choice: Mapping[str, Any],
    favors_shallow: bool,
    queries: Mapping[str, str],
    documents: Mapping[str, Mapping[str, Any]],
) -> TopicView:
    topic_id = choice["topic_id"]
    found = documents[topic_id]
    return TopicView(
        topic_id=topic_id,
        favors_shallow=favors_shallow,
        query=queries[topic_id],
        shallow_layer=_layer_of(choice["shallow"]),
        deep_layer=_layer_of(choice["deep"]),
        shallow_metrics=dict(choice["shallow"]["metrics"]),
        deep_metrics=dict(choice["deep"]["metrics"]),
        margin=choice["margin"],
        top=_documents(found["top"]),
        relevant_top=_documents(found["relevant_top"]),
        relevant_bottom=_documents(found["relevant_bottom"]),
    )


def _layer_of(result: Mapping[str, Any]) -> int:
    (layer,) = result["layers"]
    return int(layer)


def _documents(items: Sequence[Mapping[str, Any]]) -> list[DocumentView]:
    return [
        DocumentView(
            document_id=item["document_id"],
            shallow_rank=item["shallow_rank"],
            deep_rank=item["deep_rank"],
            grade=item["grade"],
        )
        for item in items
    ]
