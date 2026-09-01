"""検索トピックを読む。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hidden_subspace.corpus.archive import read_members

_TOPIC = re.compile(r"<TOPIC\s+q=(\d+)>(.*?)</TOPIC>", re.DOTALL)
_TITLE = re.compile(r"<TITLE>(.*?)</TITLE>", re.DOTALL)
_DESCRIPTION = re.compile(r"<DESCRIPTION>(.*?)</DESCRIPTION>", re.DOTALL)


@dataclass(frozen=True)
class Topic:
    """トピック 1 件。"""

    topic_id: str
    title: str
    description: str

    def __post_init__(self) -> None:
        if not self.topic_id:
            raise ValueError("topic_id が空")


def read_topics(archive: Path, members: Sequence[str]) -> list[Topic]:
    """トピックを読む。複数のファイルに分かれているため順に連結する。"""
    topics: list[Topic] = []
    for content in read_members(archive, members):
        for topic_id, body in _TOPIC.findall(content):
            topics.append(
                Topic(
                    topic_id=topic_id,
                    title=_required(_TITLE, body, "TITLE", topic_id),
                    description=_required(_DESCRIPTION, body, "DESCRIPTION", topic_id),
                )
            )
    return topics


def _required(pattern: re.Pattern[str], body: str, name: str, topic_id: str) -> str:
    found = pattern.search(body)
    if found is None:
        raise ValueError(f"トピック {topic_id} が {name} を持たない")
    return " ".join(found.group(1).split())
