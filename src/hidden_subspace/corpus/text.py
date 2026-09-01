"""レコードから検索に用いるテキストを組み立てる (ADR-0011)。"""

from __future__ import annotations

from hidden_subspace.corpus.documents import Document
from hidden_subspace.corpus.topics import Topic


def document_text(document: Document) -> str:
    """文書の検索対象テキスト。題名と抄録を繋ぐ。"""
    return f"{document.title} {document.abstract}"


def query_text(topic: Topic) -> str:
    """クエリのテキスト。トピックの題名を用いる。"""
    return topic.title
