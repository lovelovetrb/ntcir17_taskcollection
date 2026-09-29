"""図を描く。色はモデルで、線の種類は物差しで分ける。

図の文字は英語にする。環境の日本語フォントに依らずに同じ図が出るようにするため。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.lexical_correlation.records import LayerScores

MODEL_COLORS = {"bert": "#2a78d6", "simcse": "#eb6834"}
"""検証済みの 2 色 (dataviz の既定パレットの 1・2 番)。"""

YARDSTICK_LINESTYLES = {"bm25": "-", "tf-per-token": "--"}
GRID_COLOR = "#d8d7d2"
REFERENCE_COLOR = "#8a8984"


@dataclass(frozen=True)
class CorrelationSeries:
    """折れ線 1 本ぶん。層ごとの相関と、その信頼区間。"""

    model: str
    yardstick: str
    layers: tuple[int, ...]
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.model} vs {self.yardstick}"


def save_scatter_grid(
    yardstick_scores: np.ndarray,
    layer_scores: LayerScores,
    correlations: Sequence[float],
    *,
    model: str,
    yardstick: str,
    path: Path,
) -> None:
    """層ごとに 1 枚の散布図を格子に並べる。x は物差し、y はその層。題に相関係数を入れる。"""
    columns = 4
    rows = -(-len(layer_scores.layers) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
    color = MODEL_COLORS[model]

    for index, axis in enumerate(axes.flat):
        if index >= len(layer_scores.layers):
            axis.set_axis_off()
            continue
        layer = layer_scores.layers[index]
        axis.plot([0, 1], [0, 1], linestyle=":", linewidth=1, color=REFERENCE_COLOR)
        axis.scatter(
            yardstick_scores,
            layer_scores.values[index],
            s=18,
            color=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
        )
        axis.set_title(f"layer {layer} (ρ = {correlations[index]:.2f})", fontsize=10)  # noqa: RUF001
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.grid(True, color=GRID_COLOR, linewidth=0.5)
        axis.tick_params(labelsize=8)
        if index % columns == 0:
            axis.set_ylabel(f"{model} nDCG@1000", fontsize=9)
        if index // columns == rows - 1 or index + columns >= len(layer_scores.layers):
            axis.set_xlabel(f"{yardstick} nDCG@1000", fontsize=9)

    figure.suptitle(
        f"Per-topic nDCG@1000: {model} layers vs {yardstick} (n = {len(layer_scores.topic_ids)})",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def save_correlation_lines(series: Sequence[CorrelationSeries], *, path: Path) -> None:
    """層を横軸にした相関の折れ線。区間は図に入れず、表 (correlation.md) に持たせる。"""
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for item in series:
        axis.plot(
            item.layers,
            item.values,
            color=MODEL_COLORS[item.model],
            linestyle=YARDSTICK_LINESTYLES[item.yardstick],
            linewidth=2,
            marker="o",
            markersize=5,
            label=item.label,
        )

    axis.set_xlabel("layer")
    axis.set_ylabel("Spearman ρ with per-topic nDCG@1000")  # noqa: RUF001
    axis.set_xticks(list(series[0].layers))
    axis.set_ylim(0, 0.8)
    axis.grid(True, color=GRID_COLOR, linewidth=0.5)
    axis.legend(loc="lower left", fontsize=9, frameon=False)
    axis.set_title("Per-topic correlation of each layer with lexical yardsticks (n = 83)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
