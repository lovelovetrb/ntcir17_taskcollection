"""判定の一覧を、評価が受け取る形に組み直す処理を確かめる。"""

from __future__ import annotations

from hidden_subspace.corpus.judgements import Judgement
from hidden_subspace.corpus.qrels import as_qrels


def test_groups_judgements_by_topic() -> None:
    judgements = [
        Judgement("0001", "a", 2),
        Judgement("0001", "b", 0),
        Judgement("0002", "a", 1),
    ]

    assert as_qrels(judgements) == {"0001": {"a": 2, "b": 0}, "0002": {"a": 1}}


def test_keeps_every_grade_including_the_non_relevant() -> None:
    """2 値化は評価の側で行う。ここでは落とさない。"""
    assert as_qrels([Judgement("0001", "a", 0)]) == {"0001": {"a": 0}}
