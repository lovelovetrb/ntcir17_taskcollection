"""結果の記録から、トピック別の nDCG@1000 を並べた表を読む。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hidden_subspace.experiment.reader import read_topic_results

METRIC = "nDCG@1000"


@dataclass(frozen=True)
class LayerScores:
    """層ごとの実験 (layer-sweep) の記録。

    `values[i, j]` は層 `layers[i]` でのトピック `topic_ids[j]` の値。
    """

    topic_ids: tuple[str, ...]
    layers: tuple[int, ...]
    values: np.ndarray


def read_layer_scores(results_root: Path, model_id: str) -> LayerScores:
    """層ごとの記録を、層ごと・トピックごとの表にする。トピックは ID 順に並べる。"""
    by_layer: dict[int, dict[str, float]] = {}
    for result in read_topic_results(results_root, model_id, "layer-sweep"):
        (layer,) = result.configuration.layers
        by_layer.setdefault(layer, {})[result.topic_id] = result.metrics[METRIC]

    layers = tuple(sorted(by_layer))
    topic_ids = tuple(sorted(by_layer[layers[0]]))
    values = np.array([[by_layer[layer][topic_id] for topic_id in topic_ids] for layer in layers])
    return LayerScores(topic_ids=topic_ids, layers=layers, values=values)


def read_run_scores(
    results_root: Path, model_id: str, experiment: str, topic_ids: Sequence[str]
) -> np.ndarray:
    """run の評価の記録 (ADR-0023) を、指定したトピックの順に並べる。"""
    by_topic = {
        result.topic_id: result.metrics[METRIC]
        for result in read_topic_results(results_root, model_id, experiment)
    }
    missing = [topic_id for topic_id in topic_ids if topic_id not in by_topic]
    if missing:
        raise ValueError(
            f"{model_id}/{experiment} の記録にトピック {missing[:3]} がありません。"
            "層の記録と同じトピックで評価した run を使ってください。"
        )
    return np.array([by_topic[topic_id] for topic_id in topic_ids])
