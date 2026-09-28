"""実験結果の記録が、意図した形で書き出され、読み戻せることを確かめる。

`as_record` の出力はファイル形式そのものであり、名前や入れ子が変わると
書き出した結果を読む側が壊れる。
"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace.evaluation import Evaluation
from hidden_subspace.experiment.results import (
    Configuration,
    ConfigurationResult,
    DimensionChoice,
    TopicResult,
    results_of_evaluation,
)

METRICS = {"nDCG@10": 0.19, "Recall@1000": 0.30}


def configuration_of(dimensions: DimensionChoice) -> Configuration:
    return Configuration(
        model_id="cl-tohoku/bert-base-japanese-v3",
        layers=(3, 7),
        dimensions=dimensions,
        normalize_layers=True,
    )


def test_a_configuration_result_is_self_contained() -> None:
    """1 件だけ取り出しても何の結果か分かる。ディレクトリに情報を預けない。"""
    result = ConfigurationResult(
        experiment="layer-sweep",
        configuration=configuration_of(DimensionChoice(kind="all")),
        metrics=METRICS,
        zero_norm_queries=0,
        zero_norm_documents=2,
    )

    assert result.as_record() == {
        "experiment": "layer-sweep",
        "model_id": "cl-tohoku/bert-base-japanese-v3",
        "layers": [3, 7],
        "dimensions": {"kind": "all"},
        "normalize_layers": True,
        "metrics": METRICS,
        "zero_norm": {"queries": 0, "documents": 2},
    }


def test_a_topic_result_carries_the_topic_identifier() -> None:
    """ファイル名から分かるが、連結したときに区別がつくよう記録にも持たせる。"""
    result = TopicResult(
        experiment="layer-sweep",
        configuration=configuration_of(DimensionChoice(kind="all")),
        topic_id="0001",
        metrics=METRICS,
    )

    assert result.as_record()["topic_id"] == "0001"


def test_dimension_parameters_are_flattened_beside_the_kind() -> None:
    """選び方ごとに引数が異なるため、その選び方が持つものだけを書く。"""
    choice = DimensionChoice(kind="random", parameters={"count": 100, "seed": 7})

    assert choice.as_record() == {"kind": "random", "count": 100, "seed": 7}


def test_a_choice_without_parameters_writes_only_the_kind() -> None:
    assert DimensionChoice(kind="all").as_record() == {"kind": "all"}


def test_a_topic_result_is_restored_from_its_record() -> None:
    """書き出した記録を読み戻すと元に戻る。JSON では tuple が list になるため、形も戻す。"""
    result = TopicResult(
        experiment="layer-sweep",
        configuration=configuration_of(DimensionChoice(kind="all")),
        topic_id="0001",
        metrics=METRICS,
    )

    assert TopicResult.from_record(result.as_record()) == result


def test_dimension_parameters_are_gathered_back_from_beside_the_kind() -> None:
    choice = DimensionChoice(kind="random", parameters={"count": 100, "seed": 7})

    assert DimensionChoice.from_record(choice.as_record()) == choice


def test_a_choice_without_parameters_is_restored_with_no_parameters() -> None:
    choice = DimensionChoice(kind="all")

    assert DimensionChoice.from_record(choice.as_record()) == choice


def test_a_record_missing_a_field_is_rejected() -> None:
    record = TopicResult(
        experiment="layer-sweep",
        configuration=configuration_of(DimensionChoice(kind="all")),
        topic_id="0001",
        metrics=METRICS,
    ).as_record()
    del record["topic_id"]

    with pytest.raises(KeyError):
        TopicResult.from_record(record)


def test_an_evaluation_becomes_one_summary_and_one_row_per_topic() -> None:
    """指標の計算結果を記録の型に詰める。値はテンソルではなく float で入る。"""
    evaluation = Evaluation(
        topic_ids=("0001", "0002"),
        scores={"nDCG@1": torch.tensor([1.0, 0.0]), "Recall@1": torch.tensor([0.5, 0.0])},
    )

    summary, per_topic = results_of_evaluation(
        "layer-sweep", configuration_of(DimensionChoice(kind="all")), evaluation
    )

    assert summary.metrics == {"nDCG@1": 0.5, "Recall@1": 0.25}
    assert (summary.zero_norm_queries, summary.zero_norm_documents) == (0, 0)
    assert [r.topic_id for r in per_topic] == ["0001", "0002"]
    assert per_topic[0].metrics == {"nDCG@1": 1.0, "Recall@1": 0.5}
    assert all(isinstance(v, float) for r in per_topic for v in r.metrics.values())
