"""結果の書き出しを確かめる。"""

from __future__ import annotations

import json
from pathlib import Path

from hidden_subspace.experiment.results import (
    Configuration,
    ConfigurationResult,
    DimensionChoice,
    TopicResult,
)
from hidden_subspace.experiment.runner import ExperimentResults
from hidden_subspace.experiment.writer import write_results

CONFIGURATION = Configuration(
    model_id="cl-tohoku/bert-base-japanese-v3",
    layers=(3,),
    dimensions=DimensionChoice(kind="all"),
    normalize_layers=False,
)
METRICS = {"nDCG@10": 0.19}


def results_of() -> ExperimentResults:
    return ExperimentResults(
        configurations=[ConfigurationResult("layer-sweep", CONFIGURATION, METRICS)],
        topics=[
            TopicResult("layer-sweep", CONFIGURATION, "0001", METRICS),
            TopicResult("layer-sweep", CONFIGURATION, "0002", METRICS),
        ],
    )


def test_lays_out_directories_by_model_and_experiment(tmp_path: Path) -> None:
    write_results(tmp_path, results_of())

    base = tmp_path / "cl-tohoku" / "bert-base-japanese-v3" / "layer-sweep"
    assert (base / "summary.jsonl").is_file()
    assert (base / "topic" / "0001.jsonl").is_file()
    assert (base / "topic" / "0002.jsonl").is_file()


def test_writes_one_line_per_configuration(tmp_path: Path) -> None:
    write_results(tmp_path, results_of())

    path = tmp_path / "cl-tohoku" / "bert-base-japanese-v3" / "layer-sweep" / "summary.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["layers"] == [3]


def test_groups_topic_rows_into_one_file_per_topic(tmp_path: Path) -> None:
    write_results(tmp_path, results_of())

    base = tmp_path / "cl-tohoku" / "bert-base-japanese-v3" / "layer-sweep" / "topic"
    record = json.loads((base / "0001.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["topic_id"] == "0001"


def test_replaces_an_earlier_run(tmp_path: Path) -> None:
    """回し直したときに古い行が残ると、同じ構成が二重に並ぶ。"""
    write_results(tmp_path, results_of())
    write_results(tmp_path, results_of())

    path = tmp_path / "cl-tohoku" / "bert-base-japanese-v3" / "layer-sweep" / "summary.jsonl"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
