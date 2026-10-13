"""図を描く。縦軸の目盛りと色の対応は全層で揃え、層どうしを同じ物差しで比べられるようにする。

図の文字は英語にする。環境の日本語フォントに依らずに同じ図が出るようにするため。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

LINE_COLOR = "#2a78d6"
GRID_COLOR = "#d8d7d2"
REFERENCE_COLOR = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "cool_to_warm", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8c1d1d"]
)
"""dataviz の既定パレットの青と赤を両端に、灰色を 0 に置く。"""


@dataclass(frozen=True)
class RemovalDeltas:
    """基準の条件からの差。

    `values[i, j, c]` は層 `layers[i]`、トピック `topic_ids[j]` で `counts[c]` 本除いたときの値。
    """

    model: str
    metric: str
    baseline: str
    """差を取る基準の条件。`raw` なら中心化の効果を含み、`k=0` なら除去だけの効果になる。"""
    layers: tuple[int, ...]
    topic_ids: tuple[str, ...]
    counts: tuple[int, ...]
    values: np.ndarray


def save_mean_delta_grid(deltas: RemovalDeltas, *, path: Path) -> None:
    """層ごとに 1 枚、全トピック平均の差を k に沿って描く。k は等間隔に置く。"""
    means = deltas.values.mean(axis=1)
    columns = 4
    rows = -(-len(deltas.layers) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.5 * rows), sharex=True, sharey=True, squeeze=False
    )
    positions = np.arange(len(deltas.counts))
    margin = 0.05 * (means.max() - means.min())

    for index, axis in enumerate(axes.flat):
        if index >= len(deltas.layers):
            axis.set_axis_off()
            continue
        axis.axhline(0, color=REFERENCE_COLOR, linewidth=1)
        axis.plot(positions, means[index], color=LINE_COLOR, linewidth=2, marker="o", markersize=4)
        axis.set_title(f"layer {deltas.layers[index]}", loc="left", fontsize=10)
        axis.set_ylim(means.min() - margin, means.max() + margin)
        axis.set_xticks(positions, [str(count) for count in deltas.counts])
        axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5)
        axis.tick_params(labelsize=8, labelbottom=True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if index % columns == 0:
            axis.set_ylabel(f"Δ {deltas.metric} vs {deltas.baseline}", fontsize=9)
        if index + columns >= len(deltas.layers):
            axis.set_xlabel("k (0 = centering only)", fontsize=9)

    figure.suptitle(
        f"{deltas.model}: change from {deltas.baseline} after removing top-k principal components "
        f"(mean of {len(deltas.topic_ids)} topics)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def save_topic_delta_heatmaps(deltas: RemovalDeltas, *, path: Path) -> None:
    """層ごとに 1 枚、行をトピック番号順、列を k にした差のヒートマップを横に並べる。

    色の範囲は全層の絶対値の最大で 0 を中心に揃える。層ごとに揃えると、同じ色が
    層によって違う変化量を指してしまう。
    """
    limit = float(np.abs(deltas.values).max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    figure, axes = plt.subplots(
        1, len(deltas.layers), figsize=(1.6 * len(deltas.layers) + 1, 14), sharey=True
    )
    positions = np.arange(len(deltas.counts))

    for index, axis in enumerate(axes):
        axis.imshow(
            deltas.values[index], aspect="auto", cmap=DIVERGING, norm=norm, interpolation="nearest"
        )
        axis.set_title(f"layer {deltas.layers[index]}", fontsize=10)
        axis.set_xticks(positions, [str(count) for count in deltas.counts], fontsize=6, rotation=90)
        axis.set_xlabel("k", fontsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(np.arange(len(deltas.topic_ids)), list(deltas.topic_ids), fontsize=6)
    axes[0].set_ylabel("topic")

    figure.colorbar(
        ScalarMappable(norm=norm, cmap=DIVERGING),
        ax=list(axes),
        fraction=0.015,
        pad=0.01,
        label=f"Δ {deltas.metric} vs {deltas.baseline} (warm = better, cool = worse)",
    )
    figure.suptitle(
        f"{deltas.model}: per-topic change from {deltas.baseline} "
        "after removing top-k principal components",
        fontsize=12,
        y=0.91,
    )
    figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(figure)


@dataclass(frozen=True)
class CumulativeShares:
    """`values[i, c]` は層 `layers[i]` で上位 `counts[c]` 本の主成分が占める分散の割合。"""

    model: str
    layers: tuple[int, ...]
    counts: tuple[int, ...]
    values: np.ndarray


def save_cumulative_share_grid(shares: CumulativeShares, *, path: Path) -> None:
    """除いた主成分の累積の分散占有率を、mean-delta と同じ枠の並びと横軸で描く。

    並びと横軸を揃えるのは、2 枚を並べて同じ位置の枠どうしを見比べるため。
    """
    columns = 4
    rows = -(-len(shares.layers) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.5 * rows), sharex=True, sharey=True, squeeze=False
    )
    positions = np.arange(len(shares.counts))

    for index, axis in enumerate(axes.flat):
        if index >= len(shares.layers):
            axis.set_axis_off()
            continue
        axis.plot(
            positions,
            100 * shares.values[index],
            color=LINE_COLOR,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axis.set_title(f"layer {shares.layers[index]}", loc="left", fontsize=10)
        axis.set_ylim(0, 100)
        axis.set_xticks(positions, [str(count) for count in shares.counts])
        axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5)
        axis.tick_params(labelsize=8, labelbottom=True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if index % columns == 0:
            axis.set_ylabel("cumulative variance (%)", fontsize=9)
        if index + columns >= len(shares.layers):
            axis.set_xlabel("k", fontsize=9)

    figure.suptitle(
        f"{shares.model}: share of document variance in the top-k principal components",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
