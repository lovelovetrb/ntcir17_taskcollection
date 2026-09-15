"""ビューアに表示する結果を作り、ファイルに保存して読み戻す。

結果を作るにはキャッシュの読み込みと GPU が要る。一度作ったらファイルに残し、以降の起動では
それを読むだけにする (ADR-0020)。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hidden_subspace.collection import Ntcir1Paths, read_collection
from hidden_subspace.encoding.cache import load_cache
from hidden_subspace.evaluation import align_relevance
from hidden_subspace.experiment.reader import read_topic_results
from hidden_subspace.experiment.results import TopicResult
from layer_inspector.documents import RankedDocument, TopicDocuments, extract_documents
from layer_inspector.inputs import ModelInputs, decode_model_inputs
from layer_inspector.ranking import rank_selected_layers
from layer_inspector.selection import TopicChoice, TopicSelection, select_topics
from layer_inspector.verification import verify_against_records


@dataclass(frozen=True)
class InspectionSettings:
    """結果を作る条件。保存したファイルの条件と食い違えば作り直す。"""

    model_id: str
    experiment: str
    metric: str
    shallow_layers: tuple[int, ...]
    deep_layers: tuple[int, ...]
    topic_count: int
    """群ごとに選ぶトピックの数。"""
    top_count: int
    """検索結果を上から並べる件数。"""
    relevant_count: int
    """適合文書を上位・下位それぞれ並べる件数。"""


@dataclass(frozen=True)
class Inspection:
    settings: InspectionSettings
    selection: TopicSelection
    documents: dict[str, TopicDocuments]
    """トピック ID から、並べる文書への対応。"""
    inputs: ModelInputs


def build_inspection(
    settings: InspectionSettings,
    collection_root: Path,
    cache_root: Path,
    results_root: Path,
    device: str,
) -> Inspection:
    """記録からトピックを選び、検索し直して記録と照合し、表示する文書と入力を取り出す。

    検索し直した順位が記録と一致しなければ、結果を作らずに止まる。記録と同じ順位にするため、
    `device` には実験と同じく GPU を渡す。
    """
    records = read_topic_results(results_root, settings.model_id, settings.experiment)
    selection = select_topics(
        records,
        settings.metric,
        settings.shallow_layers,
        settings.deep_layers,
        settings.topic_count,
    )

    collection = read_collection(Ntcir1Paths(collection_root))
    documents = load_cache(cache_root / settings.model_id / "documents.pt").to(device)
    queries = load_cache(cache_root / settings.model_id / "queries.pt").to(device)
    rankings = rank_selected_layers(selection, documents, queries)
    relevance = align_relevance(collection.qrels, queries.item_ids, documents.item_ids)
    verify_against_records(selection, rankings, relevance.to(device))

    extracted = extract_documents(
        selection, rankings, collection.qrels, settings.top_count, settings.relevant_count
    )
    shown = {
        document.document_id
        for topic_documents in extracted.values()
        for document in (
            *topic_documents.top,
            *topic_documents.relevant_top,
            *topic_documents.relevant_bottom,
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"トークナイザを読み込めませんでした: {settings.model_id}")
    inputs = decode_model_inputs(
        [topic for topic in collection.topics if topic.topic_id in extracted],
        [document for document in collection.documents if document.document_id in shown],
        tokenizer,
    )
    return Inspection(settings=settings, selection=selection, documents=extracted, inputs=inputs)


def save_inspection(inspection: Inspection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_record_of(inspection), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_inspection(path: Path) -> Inspection:
    return _inspection_from(json.loads(path.read_text(encoding="utf-8")))


def load_or_build_inspection(
    path: Path, settings: InspectionSettings, build: Callable[[], Inspection]
) -> Inspection:
    """保存したファイルがあり、条件が同じなら読む。なければ、または条件が違えば作り直す。

    作った直後も保存したものを読み直して返す。そうしないと 1 度目だけ保存前の形のまま返り、
    2 度目以降と食い違う。
    """
    if path.is_file():
        saved = load_inspection(path)
        if saved.settings == settings:
            return saved
    save_inspection(build(), path)
    return load_inspection(path)


def _record_of(inspection: Inspection) -> dict[str, Any]:
    return {
        "settings": asdict(inspection.settings),
        "shallow_favored": [_choice_record(c) for c in inspection.selection.shallow_favored],
        "deep_favored": [_choice_record(c) for c in inspection.selection.deep_favored],
        "documents": {
            topic_id: asdict(documents) for topic_id, documents in inspection.documents.items()
        },
        "inputs": asdict(inspection.inputs),
    }


def _choice_record(choice: TopicChoice) -> dict[str, Any]:
    return {
        "topic_id": choice.topic_id,
        "shallow": choice.shallow.as_record(),
        "deep": choice.deep.as_record(),
        "margin": choice.margin,
    }


def _inspection_from(record: Mapping[str, Any]) -> Inspection:
    settings = record["settings"]
    inputs = record["inputs"]
    return Inspection(
        settings=InspectionSettings(
            model_id=settings["model_id"],
            experiment=settings["experiment"],
            metric=settings["metric"],
            shallow_layers=tuple(settings["shallow_layers"]),
            deep_layers=tuple(settings["deep_layers"]),
            topic_count=settings["topic_count"],
            top_count=settings["top_count"],
            relevant_count=settings["relevant_count"],
        ),
        selection=TopicSelection(
            shallow_favored=[_choice_from(c) for c in record["shallow_favored"]],
            deep_favored=[_choice_from(c) for c in record["deep_favored"]],
        ),
        documents={
            topic_id: _documents_from(documents)
            for topic_id, documents in record["documents"].items()
        },
        inputs=ModelInputs(queries=dict(inputs["queries"]), documents=dict(inputs["documents"])),
    )


def _choice_from(record: Mapping[str, Any]) -> TopicChoice:
    return TopicChoice(
        topic_id=record["topic_id"],
        shallow=TopicResult.from_record(record["shallow"]),
        deep=TopicResult.from_record(record["deep"]),
        margin=record["margin"],
    )


def _documents_from(record: Mapping[str, Any]) -> TopicDocuments:
    def ranked(items: list[Mapping[str, Any]]) -> list[RankedDocument]:
        return [
            RankedDocument(
                document_id=item["document_id"],
                shallow_rank=item["shallow_rank"],
                deep_rank=item["deep_rank"],
                grade=item["grade"],
            )
            for item in items
        ]

    return TopicDocuments(
        topic_id=record["topic_id"],
        top=ranked(record["top"]),
        relevant_top=ranked(record["relevant_top"]),
        relevant_bottom=ranked(record["relevant_bottom"]),
    )
