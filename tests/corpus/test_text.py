"""検索対象テキストの組み立てを確かめる。"""

from __future__ import annotations

from hidden_subspace.corpus.documents import Document
from hidden_subspace.corpus.text import document_text, query_text
from hidden_subspace.corpus.topics import Topic


def test_document_text_joins_the_title_and_abstract() -> None:
    """ADR-0011 に従い、題名と抄録を半角スペースで繋ぐ。"""
    document = Document(document_id="d1", title="題名", abstract="抄録の本文")

    assert document_text(document) == "題名 抄録の本文"


def test_query_text_uses_the_topic_title() -> None:
    """ADR-0011 に従い、クエリはトピックの題名。説明文は使わない。"""
    topic = Topic(topic_id="0001", title="ロボット", description="自律移動ロボットについて")

    assert query_text(topic) == "ロボット"
