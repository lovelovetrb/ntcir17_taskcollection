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
    results_of_evaluation,
)
from hidden_subspace.retrieval import Ranking, rank_documents
from hidden_subspace.selection import Selector
from hidden_subspace.states import HiddenStates
from hidden_subspace.vectors import SearchVectors, build_search_vectors


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
        summary, per_topic = results_of_evaluation(
            experiment,
            configuration,
            outcome.evaluation,
            zero_norm_queries=outcome.zero_norm_queries,
            zero_norm_documents=outcome.zero_norm_documents,
        )
        configurations.append(summary)
        topics.extend(per_topic)

    return ExperimentResults(configurations=configurations, topics=topics)


def rank_configuration(
    plan: ConfigurationPlan, documents: HiddenStates, queries: HiddenStates
) -> Ranking:
    """構成 1 つについて、実験と同じ手順でクエリごとに全文書を並べる。

    実験の外で順位を読むときもこれを使う。手順を書き写すと、層や次元の絞り方が
    実験と食い違っても気づけない。
    """
    return rank_documents(_vectors(queries, plan), _vectors(documents, plan))


def _evaluate_plan(
    plan: ConfigurationPlan,
    documents: HiddenStates,
    queries: HiddenStates,
    relevance: Relevance,
    ks: Sequence[int],
) -> PlanOutcome:
    query_vectors = _vectors(queries, plan)
    document_vectors = _vectors(documents, plan)
    ranking = rank_documents(query_vectors, document_vectors)
    return PlanOutcome(
        evaluation=evaluate(ranking, relevance, ks),
        zero_norm_queries=query_vectors.zero_norm_count,
        zero_norm_documents=document_vectors.zero_norm_count,
    )


def _vectors(states: HiddenStates, plan: ConfigurationPlan) -> SearchVectors:
    return build_search_vectors(_narrow(states, plan), normalize_layers=plan.normalize_layers)


def _narrow(states: HiddenStates, plan: ConfigurationPlan) -> HiddenStates:
    selected_layers = states.select_layers(plan.layers)
    return selected_layers.select_dimensions(plan.dimensions(selected_layers.dimension_indices))
