"""構成の一覧を回して結果を得る処理を確かめる。"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace.evaluation import Relevance, align_relevance
from hidden_subspace.experiment.runner import ConfigurationPlan, run_experiment
from hidden_subspace.selection import SelectAll, SelectRandom
from hidden_subspace.states import HiddenStates

Pieces = tuple[HiddenStates, HiddenStates, Relevance]

DOCUMENTS = ("a", "b", "c", "d")
TOPICS = ("0001", "0002")


@pytest.fixture
def pieces() -> Pieces:
    torch.manual_seed(0)
    documents = HiddenStates.of(torch.randn(4, 3, 8), item_ids=DOCUMENTS)
    queries = HiddenStates.of(torch.randn(2, 3, 8), item_ids=TOPICS)
    relevance = align_relevance({"0001": {"a": 2}, "0002": {"c": 1, "d": 2}}, TOPICS, DOCUMENTS)
    return documents, queries, relevance


def test_produces_one_summary_row_per_configuration(pieces: Pieces) -> None:
    documents, queries, relevance = pieces
    plans = [
        ConfigurationPlan(layers=(0,), dimensions=SelectAll(), normalize_layers=False),
        ConfigurationPlan(layers=(1,), dimensions=SelectAll(), normalize_layers=False),
    ]

    results = run_experiment(
        experiment="layer-sweep",
        model_id="m",
        documents=documents,
        queries=queries,
        relevance=relevance,
        plans=plans,
        ks=(1, 2),
    )

    assert len(results.configurations) == 2
    assert [r.configuration.layers for r in results.configurations] == [(0,), (1,)]


def test_produces_one_topic_row_per_configuration_and_topic(pieces: Pieces) -> None:
    documents, queries, relevance = pieces
    plans = [ConfigurationPlan(layers=(0,), dimensions=SelectAll(), normalize_layers=False)]

    results = run_experiment(
        experiment="layer-sweep",
        model_id="m",
        documents=documents,
        queries=queries,
        relevance=relevance,
        plans=plans,
        ks=(1,),
    )

    assert len(results.topics) == len(TOPICS)
    assert {r.topic_id for r in results.topics} == set(TOPICS)


def test_summary_is_the_average_of_the_topic_rows(pieces: Pieces) -> None:
    documents, queries, relevance = pieces
    plans = [ConfigurationPlan(layers=(0,), dimensions=SelectAll(), normalize_layers=False)]

    results = run_experiment(
        experiment="layer-sweep",
        model_id="m",
        documents=documents,
        queries=queries,
        relevance=relevance,
        plans=plans,
        ks=(1,),
    )

    summary = results.configurations[0].metrics["nDCG@1"]
    per_topic = [r.metrics["nDCG@1"] for r in results.topics]
    assert summary == pytest.approx(sum(per_topic) / len(per_topic))


def test_records_the_dimension_choice(pieces: Pieces) -> None:
    documents, queries, relevance = pieces
    plans = [
        ConfigurationPlan(
            layers=(0, 1), dimensions=SelectRandom(count=4, seed=3), normalize_layers=True
        )
    ]

    results = run_experiment(
        experiment="random-dimensions",
        model_id="m",
        documents=documents,
        queries=queries,
        relevance=relevance,
        plans=plans,
        ks=(1,),
    )

    record = results.configurations[0].as_record()
    assert record["dimensions"] == {"kind": "random", "count": 4, "seed": 3}
    assert record["layers"] == [0, 1]
    assert record["normalize_layers"] is True
    assert record["experiment"] == "random-dimensions"
