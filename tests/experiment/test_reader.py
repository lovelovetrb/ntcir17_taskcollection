"""書き出した結果の読み込みを確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hidden_subspace.experiment.reader import read_topic_results
from hidden_subspace.experiment.results import (
    Configuration,
    ConfigurationResult,
    DimensionChoice,
    TopicResult,
)
from hidden_subspace.experiment.runner import ExperimentResults
from hidden_subspace.experiment.writer import write_results

MODEL_ID = "cl-tohoku/bert-base-japanese-v3"
EXPERIMENT = "layer-sweep"


def topic_result_of(topic_id: str, layer: int) -> TopicResult:
    configuration = Configuration(
        model_id=MODEL_ID,
        layers=(layer,),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
    )
    return TopicResult(EXPERIMENT, configuration, topic_id, {"nDCG@1000": 0.1 * layer})


def write_topics(root: Path, topics: list[TopicResult]) -> None:
    summary = ConfigurationResult(EXPERIMENT, topics[0].configuration, {"nDCG@1000": 0.0}, 0, 0)
    write_results(root, ExperimentResults(configurations=[summary], topics=topics))


def test_reads_every_row_of_every_topic_file_in_file_name_order(tmp_path: Path) -> None:
    """ファイルの並びはファイルシステムに任せると揺れる。名前順に読み、行の順は保つ。"""
    write_topics(
        tmp_path,
        [
            topic_result_of("0002", 0),
            topic_result_of("0001", 0),
            topic_result_of("0002", 1),
            topic_result_of("0001", 1),
        ],
    )

    assert read_topic_results(tmp_path, MODEL_ID, EXPERIMENT) == [
        topic_result_of("0001", 0),
        topic_result_of("0001", 1),
        topic_result_of("0002", 0),
        topic_result_of("0002", 1),
    ]


def test_rejects_an_experiment_that_has_not_been_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"run_experiment\.py"):
        read_topic_results(tmp_path, MODEL_ID, EXPERIMENT)


def test_rejects_a_directory_without_records(tmp_path: Path) -> None:
    """記録のないディレクトリは、実験が途中で止まった跡である可能性が高い。"""
    (tmp_path / MODEL_ID / EXPERIMENT / "topic").mkdir(parents=True)

    with pytest.raises(ValueError):
        read_topic_results(tmp_path, MODEL_ID, EXPERIMENT)
