"""NTCIR-1 または NTCIR-2 のコレクション一式をまとめて読む。

配布物のどのファイルがどの役割かは固定されているため、その対応をここに閉じ込める
(ADR-0009, ADR-0026)。どちらのコレクションも同じ `Collection` の形に落とし込み、
評価の側はコレクションの違いを知らない。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import assert_never

from hidden_subspace.corpus.documents import Document, as_ntcir2_document_id, read_documents
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
        "mlir/rel2_ntc1-j1_0001-0030",
        "mlir/rel2_ntc1-j1_0031-0083",
    )


@dataclass(frozen=True)
class Ntcir2Paths:
    """配布された NTCIR-2 のディレクトリと、その中の役割の対応。

    NTCIR-2 の文書集合は NTCIR-1 の文書を含むが、NTCIR-2 の配布物には入っていない。
    そのため NTCIR-1 のディレクトリも受け取る。
    """

    ntcir1_root: Path
    ntcir2_root: Path
    ntcir1_documents_archive: str = "MLIR.TGZ"
    ntcir1_documents_member: str = "mlir/ntc1-j1"
    documents_archive: str = "j-docs.tgz"
    gakkai_member: str = "j-docs/ntc2-j1g"
    kaken_member: str = "j-docs/ntc2-j1k"
    topics_archive: str = "topics.tgz"
    topics_members: tuple[str, ...] = ("topics/topic-j0101-0149",)
    qrels_archive: str = "rels.tgz"
    qrels_members: tuple[str, ...] = ("rels/rel2_ntc2-j2_0101-0149",)


CollectionPaths = Ntcir1Paths | Ntcir2Paths

COLLECTION_NAMES = ("ntcir1", "ntcir2")
"""実行時に選ぶコレクションの名前。キャッシュと記録のディレクトリ名にも使う (ADR-0026)。"""


def collection_paths(name: str, data_root: Path) -> CollectionPaths:
    """配布物を置いたディレクトリの下で、名前に対応するコレクションのパスを返す。"""
    match name:
        case "ntcir1":
            return Ntcir1Paths(data_root / "NTCIR-1")
        case "ntcir2":
            return Ntcir2Paths(data_root / "NTCIR-1", data_root / "NTCIR-2")
        case _:
            raise ValueError(
                f"コレクション {name} はありません。"
                f"{', '.join(COLLECTION_NAMES)} から選んでください。"
            )


@dataclass(frozen=True)
class Collection:
    """検索の対象と、その答え合わせ。"""

    documents: list[Document]
    topics: list[Topic]
    qrels: dict[str, dict[str, int]]


def read_collection(paths: CollectionPaths) -> Collection:
    """文書・トピック・判定を読む。どのコレクションを読むかは paths の型で決まる。"""
    match paths:
        case Ntcir1Paths():
            return _read_ntcir1(paths)
        case Ntcir2Paths():
            return _read_ntcir2(paths)
        case _:
            assert_never(paths)


def _read_ntcir1(paths: Ntcir1Paths) -> Collection:
    documents_archive = paths.root / paths.documents_archive
    return Collection(
        documents=read_documents(documents_archive, paths.documents_member),
        topics=read_topics(paths.root / paths.topics_archive, paths.topics_members),
        qrels=as_qrels(read_judgements(documents_archive, paths.qrels_members)),
    )


def _read_ntcir2(paths: Ntcir2Paths) -> Collection:
    """文書は NTCIR-1 の部分、学会発表、科研費報告書の順に連結する。

    組織者が作る `ntc12-j1gk.mod` と同じ順である。NTCIR-1 の部分は ID を NTCIR-2 の
    qrels が参照する形に書き換え、科研費報告書は研究課題名 PJNM を題名とする (ADR-0026)。
    """
    ntcir1_part = [
        replace(document, document_id=as_ntcir2_document_id(document.document_id))
        for document in read_documents(
            paths.ntcir1_root / paths.ntcir1_documents_archive, paths.ntcir1_documents_member
        )
    ]
    documents_archive = paths.ntcir2_root / paths.documents_archive
    gakkai = read_documents(documents_archive, paths.gakkai_member)
    kaken = read_documents(documents_archive, paths.kaken_member, title_tag="PJNM")
    return Collection(
        documents=ntcir1_part + gakkai + kaken,
        topics=read_topics(paths.ntcir2_root / paths.topics_archive, paths.topics_members),
        qrels=as_qrels(
            read_judgements(paths.ntcir2_root / paths.qrels_archive, paths.qrels_members)
        ),
    )
