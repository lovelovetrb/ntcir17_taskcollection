"""判定の記録から、トピックと側ごとに 4 区分の件数を数える (issue #46)。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.word_sense.judge import Category
from analysis.word_sense.records import SIDES, judgement_path, read_judgements

CATEGORIES = tuple(Category)


@dataclass(frozen=True)
class Distribution:
    """`counts[side][i, j]` はトピック `topic_ids[i]` で区分 `CATEGORIES[j]` に判定された件数。"""

    topic_ids: tuple[str, ...]
    titles: tuple[str, ...]
    counts: dict[str, np.ndarray]

    def proportions(self, side: str) -> np.ndarray:
        """トピックごとの割合。各行の和は 1。"""
        counts = self.counts[side]
        return np.asarray(counts / counts.sum(axis=1, keepdims=True), dtype=float)

    def overall(self, side: str) -> np.ndarray:
        """全トピックをまとめた割合。"""
        total = self.counts[side].sum(axis=0)
        return np.asarray(total / total.sum(), dtype=float)


def read_distribution(root: Path, model: str) -> Distribution:
    """両側の記録が揃っているトピックだけを、トピック番号の順に数える。"""
    topic_ids = tuple(
        sorted(
            directory.name
            for directory in (root / model).iterdir()
            if all(judgement_path(root, model, directory.name, side).is_file() for side in SIDES)
        )
    )
    if not topic_ids:
        raise FileNotFoundError(
            f"{root / model} に判定の記録がありません。先に collect_judgements を実行してください。"
        )

    counts = {side: np.zeros((len(topic_ids), len(CATEGORIES)), dtype=int) for side in SIDES}
    titles: list[str] = []
    for i, topic_id in enumerate(topic_ids):
        by_side = {
            side: read_judgements(judgement_path(root, model, topic_id, side)) for side in SIDES
        }
        for side, judgements in by_side.items():
            for judgement in judgements:
                counts[side][i, CATEGORIES.index(Category(judgement.llm.category))] += 1
        titles.append(by_side[SIDES[0]][0].title)
    return Distribution(topic_ids=topic_ids, titles=tuple(titles), counts=counts)
