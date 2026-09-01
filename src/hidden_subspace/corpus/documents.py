"""文書レコードを読む。

元のレコードに近い形で保持する。検索対象テキストの組み立ては別に行う。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hidden_subspace.corpus.archive import read_members

_FIELD = re.compile(r"<(ACCN|TITL|ABST)[^>]*>(.*?)</\1>", re.DOTALL)
_INNER_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Document:
    """文書 1 件。"""

    document_id: str
    title: str
    abstract: str

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id が空")


def read_documents(archive: Path, member: str) -> list[Document]:
    """文書集合を読む。

    対象とする文書集合は全件が題名と抄録を持つ。欠けている場合は別の文書集合を
    読んでいる可能性があるため、黙って飛ばさずに拒否する。
    """
    (content,) = read_members(archive, (member,))
    documents: list[Document] = []
    for record in content.split("<REC>")[1:]:
        fields = {name: _INNER_TAG.sub(" ", body) for name, body in _FIELD.findall(record)}
        for required in ("ACCN", "TITL", "ABST"):
            if required not in fields:
                raise ValueError(f"{required} を持たないレコードがある: {record[:80]!r}")
        documents.append(
            Document(
                document_id=fields["ACCN"].strip(),
                title=_collapse(fields["TITL"]),
                abstract=_collapse(fields["ABST"]),
            )
        )
    return documents


def _collapse(text: str) -> str:
    return " ".join(text.split())
