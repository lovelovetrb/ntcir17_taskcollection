"""判定の一覧を、評価が受け取る形に組み直す。"""

from __future__ import annotations

from collections.abc import Sequence

from hidden_subspace.corpus.judgements import Judgement


def as_qrels(judgements: Sequence[Judgement]) -> dict[str, dict[str, int]]:
    """トピック ID から、文書 ID と grade の対応への写像にする。

    2 値化は評価の側で行うため、非適合の判定も落とさない。
    """
    qrels: dict[str, dict[str, int]] = {}
    for judgement in judgements:
        qrels.setdefault(judgement.topic_id, {})[judgement.document_id] = judgement.grade
    return qrels
