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


@dataclass(frozen=True)
class PlanOutcome:
    """構成 1 つを回して分かったこと。"""

    evaluation: Evaluation
    zero_norm_queries: int
    zero_norm_documents: int


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
        outcome = _evaluate_plan(plan, documents, queries, relevance, ks)
        evaluation = outcome.evaluation

        configurations.append(
            ConfigurationResult(
                experiment=experiment,
                configuration=configuration,
                metrics=evaluation.macro_average(),
                zero_norm_queries=outcome.zero_norm_queries,
                zero_norm_documents=outcome.zero_norm_documents,
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
) -> PlanOutcome:
    query_vectors = build_search_vectors(
        _narrow(queries, plan), normalize_layers=plan.normalize_layers
    )
    document_vectors = build_search_vectors(
        _narrow(documents, plan), normalize_layers=plan.normalize_layers
    )
    ranking = rank_documents(query_vectors, document_vectors)
    return PlanOutcome(
        evaluation=evaluate(ranking, relevance, ks),
        zero_norm_queries=query_vectors.zero_norm_count,
        zero_norm_documents=document_vectors.zero_norm_count,
    )


def _narrow(states: HiddenStates, plan: ConfigurationPlan) -> HiddenStates:
    selected_layers = states.select_layers(plan.layers)
    return selected_layers.select_dimensions(plan.dimensions(selected_layers.dimension_indices))
