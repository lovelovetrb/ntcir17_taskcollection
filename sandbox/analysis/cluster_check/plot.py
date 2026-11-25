"""シルエット係数を、層ごとに 1 枚、k に沿った折れ線で描く (ADR-0037)。

文書の値を線で、乱数データの値を回の平均の線と最小から最大の帯で描く。縦軸は
シルエット係数のとりうる範囲 (-1〜1) に固定し、0 に線を引く。図の文字は英語にする。

    PYTHONPATH=sandbox uv run python -m analysis.cluster_check.plot \
        --collection ntcir2 --model cl-tohoku/bert-base-japanese-v3 --pooling mean
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.cluster_check.records import DOCUMENTS, GAUSSIAN, SilhouetteRecord, read_records
from analysis.cluster_check.run import output_dir_of
from hidden_subspace.collection import COLLECTION_NAMES
from hidden_subspace.encoding.pooling import MEAN, POOLINGS

DOCUMENT_COLOR = "#2a78d6"
GAUSSIAN_COLOR = "#8a8984"
REFERENCE_COLOR = "#52514e"
GRID_COLOR = "#d8d7d2"
COLUMNS = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTION_NAMES)
    parser.add_argument("--model", required=True, help="モデルの識別子")
    parser.add_argument("--pooling", default=MEAN, choices=POOLINGS)
    return parser.parse_args()


def save_silhouette_grid(records: list[SilhouetteRecord], *, title: str, path: Path) -> None:
    by_layer: dict[int, list[SilhouetteRecord]] = defaultdict(list)
    for record in records:
        by_layer[record.layer].append(record)
    layers = sorted(by_layer)
    ks = sorted({record.k for record in records})
    positions = np.arange(len(ks))
    rows = -(-len(layers) // COLUMNS)
    figure, axes = plt.subplots(
        rows, COLUMNS, figsize=(3.2 * COLUMNS, 2.5 * rows), sharex=True, sharey=True, squeeze=False
    )

    for index, axis in enumerate(axes.flat):
        if index >= len(layers):
            axis.set_axis_off()
            continue
        layer_records = by_layer[layers[index]]
        documents = [_value(layer_records, DOCUMENTS, k) for k in ks]
        gaussian = np.array([_values(layer_records, GAUSSIAN, k) for k in ks])

        axis.axhline(0, color=REFERENCE_COLOR, linewidth=1)
        axis.fill_between(
            positions, gaussian.min(axis=1), gaussian.max(axis=1), color=GAUSSIAN_COLOR, alpha=0.3
        )
        axis.plot(
            positions,
            gaussian.mean(axis=1),
            color=GAUSSIAN_COLOR,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Gaussian (same covariance)",
        )
        axis.plot(
            positions,
            documents,
            color=DOCUMENT_COLOR,
            linewidth=2,
            marker="o",
            markersize=4,
            label="documents",
        )
        axis.set_title(f"layer {layers[index]}", loc="left", fontsize=10)
        axis.set_ylim(-1, 1)
        axis.set_xticks(positions, [str(k) for k in ks])
        axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5)
        axis.tick_params(labelsize=8, labelbottom=True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if index % COLUMNS == 0:
            axis.set_ylabel("mean silhouette", fontsize=9)
        if index + COLUMNS >= len(layers):
            axis.set_xlabel("k", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", fontsize=9, frameon=False)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _value(records: list[SilhouetteRecord], source: str, k: int) -> float:
    (value,) = _values(records, source, k)
    return value


def _values(records: list[SilhouetteRecord], source: str, k: int) -> list[float]:
    values = [r.silhouette for r in records if r.source == source and r.k == k]
    if not values:
        raise ValueError(f"{source} の k={k} の記録がありません。run を回し直してください。")
    return values


def main() -> int:
    arguments = parse_arguments()
    directory = output_dir_of(arguments.collection, arguments.model, arguments.pooling)
    records = read_records(directory)
    repeats = len({r.repeat for r in records if r.source == GAUSSIAN})
    sample = records[0].sample_size
    path = directory / "silhouette.png"
    save_silhouette_grid(
        records,
        title=(
            f"{arguments.model} ({arguments.collection}, {arguments.pooling}): mean silhouette of "
            f"{sample:,} documents vs Gaussian with the same covariance "
            f"({repeats} draws, band = min-max)"
        ),
        path=path,
    )
    print(f"書き出し {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
