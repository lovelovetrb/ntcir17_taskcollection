"""run.py の記録から、生と比べた nDCG@1000 の変化を図にする (issue #47)。

全トピック平均の変化を層ごとの折れ線に、トピックごとの変化を層ごとのヒートマップに
描き、output/ に置く。GPU は使わない。

    PYTHONPATH=sandbox uv run python -m analysis.pc_removal.plot
"""

from __future__ import annotations

import numpy as np

from analysis.pc_removal.figures import (
    RemovalDeltas,
    save_mean_delta_grid,
    save_topic_delta_heatmaps,
)
from analysis.pc_removal.records import RAW, TopicRecord, read_records, removal_condition
from analysis.pc_removal.run import MODELS, OUTPUT_DIR, REMOVED_COUNTS

METRIC = "nDCG@1000"


def removal_deltas(records: list[TopicRecord], *, model: str) -> RemovalDeltas:
    """記録を層・トピック・k の差の表にする。トピックは ID 順に並べる。"""
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
                    - scores[(layer, RAW, topic_id)]
                    for count in REMOVED_COUNTS
                ]
                for topic_id in topic_ids
            ]
            for layer in layers
        ]
    )
    return RemovalDeltas(
        model=model,
        metric=METRIC,
        layers=layers,
        topic_ids=topic_ids,
        counts=REMOVED_COUNTS,
        values=values,
    )


def main() -> None:
    for model in MODELS:
        deltas = removal_deltas(read_records(OUTPUT_DIR / f"metrics-{model}.jsonl"), model=model)
        save_mean_delta_grid(deltas, path=OUTPUT_DIR / f"mean-delta-{model}.png")
        save_topic_delta_heatmaps(deltas, path=OUTPUT_DIR / f"topic-delta-{model}.png")


if __name__ == "__main__":
    main()
