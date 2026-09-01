"""適合性判定を読む。

1 行 1 判定のまま読む。トピック別の写像への組み直しは使う側で行う。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hidden_subspace.corpus.archive import read_members

GRADE_VALUES = {"A": 2, "B": 1, "C": 0}
"""A を適合、B を部分的に適合、C を非適合とする配布データの表記を数値にする。"""


@dataclass(frozen=True)
class Judgement:
    """トピックと文書の組に対する判定 1 件。"""

    topic_id: str
    document_id: str
    grade: int


def read_judgements(archive: Path, members: Sequence[str]) -> list[Judgement]:
    """判定を読む。

    1 行は TAB 区切りで (トピック, 判定, 文書, 2 値フラグ[, 判定者コメント])。
    コメントの有無で列数が 4 と 5 で揺れるため、先頭 3 列だけを使う。
    """
    judgements: list[Judgement] = []
    for content in read_members(archive, members):
        for line in content.splitlines():
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            topic_id, grade, document_id = fields[0], fields[1], fields[2]
            if grade not in GRADE_VALUES:
                raise ValueError(f"未知の判定: {grade!r} ({line[:60]!r})")
            judgements.append(
                Judgement(
                    topic_id=topic_id,
                    document_id=document_id,
                    grade=GRADE_VALUES[grade],
                )
            )
    return judgements
