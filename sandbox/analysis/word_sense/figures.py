"""4 区分の分布を積み上げ棒グラフにする。

図の文字は英語にする。トピックの題名だけは日本語なので、目盛りのラベルに日本語フォントを使う。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.word_sense.distribution import CATEGORIES, Distribution
from analysis.word_sense.judge import Category

LABELS = {
    Category.MATCH: "match",
    Category.PARTIAL_MATCH: "partial match",
    Category.OTHER_SENSE: "other sense",
    Category.UNRELATED: "unrelated",
}
COLORS = {
    Category.MATCH: "#2a78d6",
    Category.PARTIAL_MATCH: "#eb6834",
    Category.OTHER_SENSE: "#1baf7a",
    Category.UNRELATED: "#eda100",
}
"""検証済みの 4 色 (dataviz の既定パレットの 1〜4 番)。区分の順に固定する。"""

JAPANESE_FONTS = ["DejaVu Sans", "Noto Sans CJK JP", "IPAexGothic"]
"""欧文は DejaVu Sans、日本語はその次に見つかったフォントで描く。

日本語フォントが無い環境では題名が豆腐になる。
"""

SIDE_LABELS = {"shallow": "shallow (best of layers 1-2)", "deep": "deep (best of layers 11-12)"}
GRID_COLOR = "#d8d7d2"


def _stack(axis: plt.Axes, positions: np.ndarray, proportions: np.ndarray, width: float) -> None:
    """区分の順に下から積む。区切りは白い細線にして、隣り合う色の境目を見えるようにする。"""
    bottom = np.zeros(len(positions))
    for j, category in enumerate(CATEGORIES):
        axis.bar(
            positions,
            proportions[:, j],
            width,
            bottom=bottom,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.6,
            label=LABELS[category],
        )
        bottom += proportions[:, j]


def save_overall(distribution: Distribution, *, model: str, path: Path) -> None:
    """浅い側と深い側の 2 本の積み上げ棒。各区分の割合を棒の中に書く。"""
    sides = list(distribution.counts)
    proportions = np.array([distribution.overall(side) for side in sides])
    figure, axis = plt.subplots(figsize=(6, 5))
    positions = np.arange(len(sides))
    _stack(axis, positions, proportions, width=0.55)

    for i in range(len(sides)):
        bottom = 0.0
        for j in range(len(CATEGORIES)):
            share = proportions[i, j]
            if share >= 0.03:
                axis.text(
                    i, bottom + share / 2, f"{share:.0%}", ha="center", va="center", fontsize=9
                )
            bottom += share

    axis.set_xticks(positions, [SIDE_LABELS[side] for side in sides], fontsize=9)
    axis.set_ylim(0, 1)
    axis.set_ylabel("share of top-100 documents")
    axis.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    total = int(distribution.counts[sides[0]].sum())
    axis.set_title(
        f"{model}: word-sense judgements of top-100 documents\n(n = {total} per side)", fontsize=11
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def save_by_topic(distribution: Distribution, *, model: str, path: Path) -> None:
    """横軸をトピック、縦軸を割合にした積み上げ棒。浅い側を上段、深い側を下段に並べる。"""
    sides = list(distribution.counts)
    positions = np.arange(len(distribution.topic_ids))
    figure, axes = plt.subplots(
        len(sides), 1, figsize=(max(12, 0.2 * len(positions)), 10), sharex=True
    )
    for axis, side in zip(axes, sides, strict=True):
        _stack(axis, positions, distribution.proportions(side), width=0.8)
        axis.set_xlim(-0.6, len(positions) - 0.4)
        axis.set_ylim(0, 1)
        axis.set_ylabel(SIDE_LABELS[side], fontsize=9)
        axis.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        axis.set_axisbelow(True)

    labels = [
        f"{topic_id} {title}"
        for topic_id, title in zip(distribution.topic_ids, distribution.titles, strict=True)
    ]
    # 上から下に読む向きにして、軸のすぐ下に ID が来るようにする
    axes[-1].set_xticks(positions, labels, rotation=-90, fontsize=7, fontfamily=JAPANESE_FONTS)
    axes[-1].set_xlabel("topic")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(CATEGORIES), frameon=False)
    figure.suptitle(f"{model}: word-sense judgements of top-100 documents per topic")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
