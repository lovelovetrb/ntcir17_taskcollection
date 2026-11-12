"""run.py の記録から、生と比べた nDCG@1000 の変化を図にする (issue #47)。

全トピック平均の変化を層ごとの折れ線に、トピックごとの変化を層ごとのヒートマップに
描く。ヒートマップは、生を基準にしたものと、中心化だけをした k=0 を基準にしたものの
2 枚を描く。除いた主成分の累積の分散占有率も、平均の変化と同じ並びの折れ線にする。記録は
`output/<--collection の値>/<model_id>/mean/` から読み、図も同じ場所に置く。GPU は使わない。

    PYTHONPATH=sandbox uv run python -m analysis.pc_removal.plot --collection ntcir1
"""

from __future__ import annotations

import argparse

import numpy as np

from analysis.pc_removal.figures import (
    CumulativeShares,
    RemovalDeltas,
    save_cumulative_share_grid,
    save_mean_delta_grid,
    save_topic_delta_heatmaps,
)
from analysis.pc_removal.records import (
    RAW,
    TopicRecord,
    VarianceRecord,
    read_records,
    read_variance_records,
    removal_condition,
)
from analysis.pc_removal.run import COLLECTIONS, MODELS, REMOVED_COUNTS, output_dir_of

METRIC = "nDCG@1000"


def removal_deltas(
    records: list[TopicRecord], *, model: str, baseline: str, counts: tuple[int, ...]
) -> RemovalDeltas:
    """記録を、`baseline` の条件からの層・トピック・k の差の表にする。トピックは ID 順に並べる。"""
    scores = {
        (record.layer, record.condition, record.topic_id): record.metrics[METRIC]
        for record in records
    }
    layers = tuple(sorted({record.layer for record in records}))
    topic_ids = tuple(sorted({record.topic_id for record in records}))
    values = np.array(
        [
            [
                [
                    scores[(layer, removal_condition(count), topic_id)]
                    - scores[(layer, baseline, topic_id)]
                    for count in counts
                ]
                for topic_id in topic_ids
            ]
            for layer in layers
        ]
    )
    return RemovalDeltas(
        model=model,
        metric=METRIC,
        baseline=baseline,
        layers=layers,
        topic_ids=topic_ids,
        counts=counts,
        values=values,
    )


def cumulative_shares(records: list[VarianceRecord], *, model: str) -> CumulativeShares:
    shares = {(record.layer, record.count): record.cumulative_share for record in records}
    layers = tuple(sorted({record.layer for record in records}))
    return CumulativeShares(
        model=model,
        layers=layers,
        counts=REMOVED_COUNTS,
        values=np.array([[shares[(layer, count)] for count in REMOVED_COUNTS] for layer in layers]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    collection = parser.parse_args().collection

    for model, model_id in MODELS.items():
        output_dir = output_dir_of(collection, model_id)
        records = read_records(output_dir / "metrics.jsonl")
        deltas = removal_deltas(records, model=model, baseline=RAW, counts=REMOVED_COUNTS)
        save_mean_delta_grid(deltas, path=output_dir / "mean-delta.png")
        save_topic_delta_heatmaps(deltas, path=output_dir / "topic-delta.png")
        # k=0 を基準にすると、中心化の効果を除いた、主成分を除くこと自体の効果になる
        centered = removal_deltas(
            records, model=model, baseline=removal_condition(0), counts=REMOVED_COUNTS[1:]
        )
        save_topic_delta_heatmaps(centered, path=output_dir / "topic-delta-centered.png")
        shares = cumulative_shares(
            read_variance_records(output_dir / "variance.jsonl"), model=model
        )
        save_cumulative_share_grid(shares, path=output_dir / "variance.png")


if __name__ == "__main__":
    main()
