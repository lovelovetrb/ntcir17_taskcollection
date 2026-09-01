"""NTCIR-1 のコレクション一式をまとめて読む。

配布物のどのファイルがどの役割かは固定されているため、その対応をここに閉じ込める
(ADR-0009)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hidden_subspace.corpus.documents import Document, read_documents
from hidden_subspace.corpus.judgements import read_judgements
from hidden_subspace.corpus.qrels import as_qrels
from hidden_subspace.corpus.topics import Topic, read_topics


@dataclass(frozen=True)
class Ntcir1Paths:
    """配布された NTCIR-1 のディレクトリと、その中の役割の対応。"""

    root: Path
    documents_archive: str = "MLIR.TGZ"
    documents_member: str = "mlir/ntc1-j1"
    topics_archive: str = "TOPICS.TGZ"
    topics_members: tuple[str, ...] = (
        "topics/topic0001-0030",
        "topics/topic0031-0083",
    )
    qrels_members: tuple[str, ...] = (
        "mlir/rel1_ntc1-j1_0001-0030",
        "mlir/rel1_ntc1-j1_0031-0083",
    )


@dataclass(frozen=True)
class Collection:
    """検索の対象と、その答え合わせ。"""

    documents: list[Document]
    topics: list[Topic]
    qrels: dict[str, dict[str, int]]


def read_collection(paths: Ntcir1Paths) -> Collection:
    """文書・トピック・判定を読む。"""
    documents_archive = paths.root / paths.documents_archive
    return Collection(
        documents=read_documents(documents_archive, paths.documents_member),
        topics=read_topics(paths.root / paths.topics_archive, paths.topics_members),
        qrels=as_qrels(read_judgements(documents_archive, paths.qrels_members)),
    )
