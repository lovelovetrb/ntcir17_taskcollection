"""構成の一覧を回して、指標を集める。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hidden_subspace.evaluation import Evaluation, Relevance, evaluate
from hidden_subspace.experiment.describe import describe_dimensions
from hidden_subspace.experiment.results import (
    Configuration,
    ConfigurationResult,
    TopicResult,
)
from hidden_subspace.retrieval import rank_documents
from hidden_subspace.selection import Selector
from hidden_subspace.states import HiddenStates
from hidden_subspace.vectors import build_search_vectors


@dataclass(frozen=True)
class ConfigurationPlan:
    """1 つの構成をどう作るか。

    層は使う番号を直接指定する。次元は選び方を渡し、記録用の表現はそこから導く。
    """

    layers: tuple[int, ...]
    dimensions: Selector
    normalize_layers: bool


@dataclass(frozen=True)
class ExperimentResults:
    """構成ごとの結果と、トピックごとの結果。"""

    configurations: list[ConfigurationResult]
    topics: list[TopicResult]


def run_experiment(
    *,
    experiment: str,
    model_id: str,
    documents: HiddenStates,
    queries: HiddenStates,
    relevance: Relevance,
    plans: Sequence[ConfigurationPlan],
    ks: Sequence[int],
) -> ExperimentResults:
    """構成ごとに検索して評価する。"""
    configurations: list[ConfigurationResult] = []
    topics: list[TopicResult] = []

    for plan in plans:
        configuration = Configuration(
            model_id=model_id,
            layers=plan.layers,
            dimensions=describe_dimensions(plan.dimensions),
            normalize_layers=plan.normalize_layers,
        )
        evaluation = _evaluate_plan(plan, documents, queries, relevance, ks)

        configurations.append(
            ConfigurationResult(
                experiment=experiment,
                configuration=configuration,
                metrics=evaluation.macro_average(),
            )
        )
        for row, topic_id in enumerate(evaluation.topic_ids):
            topics.append(
                TopicResult(
                    experiment=experiment,
                    configuration=configuration,
                    topic_id=topic_id,
                    metrics={
                        name: float(values[row].item())
                        for name, values in evaluation.scores.items()
                    },
                )
            )

    return ExperimentResults(configurations=configurations, topics=topics)


def _evaluate_plan(
    plan: ConfigurationPlan,
    documents: HiddenStates,
    queries: HiddenStates,
    relevance: Relevance,
    ks: Sequence[int],
) -> Evaluation:
    narrowed_documents = _narrow(documents, plan)
    narrowed_queries = _narrow(queries, plan)
    ranking = rank_documents(
        build_search_vectors(narrowed_queries, normalize_layers=plan.normalize_layers),
        build_search_vectors(narrowed_documents, normalize_layers=plan.normalize_layers),
    )
    return evaluate(ranking, relevance, ks)


def _narrow(states: HiddenStates, plan: ConfigurationPlan) -> HiddenStates:
    selected_layers = states.select_layers(plan.layers)
    return selected_layers.select_dimensions(plan.dimensions(selected_layers.dimension_indices))
