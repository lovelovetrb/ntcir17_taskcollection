"""文書レコードを読む。

元のレコードに近い形で保持する。検索対象テキストの組み立ては別に行う。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hidden_subspace.corpus.archive import read_members

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


def read_documents(archive: Path, member: str, *, title_tag: str = "TITL") -> list[Document]:
    """文書集合を読む。

    対象とする文書集合は全件が題名と抄録を持つ。欠けている場合は別の文書集合を
    読んでいる可能性があるため、黙って飛ばさずに拒否する。

    題名のタグは文書集合によって異なる。kaken は TITL を持たず、研究課題名の PJNM を
    題名とする (ADR-0026)。
    """
    field = re.compile(rf"<(ACCN|{re.escape(title_tag)}|ABST)[^>]*>(.*?)</\1>", re.DOTALL)
    (content,) = read_members(archive, (member,))
    documents: list[Document] = []
    for record in content.split("<REC>")[1:]:
        fields = {name: _INNER_TAG.sub(" ", body) for name, body in field.findall(record)}
        for required in ("ACCN", title_tag, "ABST"):
            if required not in fields:
                raise ValueError(f"{required} を持たないレコードがある: {record[:80]!r}")
        documents.append(
            Document(
                document_id=fields["ACCN"].strip(),
                title=_collapse(fields[title_tag]),
                abstract=_collapse(fields["ABST"]),
            )
        )
    return documents


def _collapse(text: str) -> str:
    return " ".join(text.split())
