"""検索結果の上位と適合文書を、両方の層での順位と判定つきで取り出す。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from hidden_subspace.evaluation import RELEVANT_FROM, Qrels
from hidden_subspace.experiment.results import TopicResult
from hidden_subspace.retrieval import Ranking
from layer_inspector.selection import TopicChoice, TopicSelection


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    shallow_rank: int
    """浅い側の層での順位。1 始まり。"""
    deep_rank: int
    """深い側の層での順位。1 始まり。"""
    grade: int | None
    """qrels の grade。判定を受けていない文書は None。"""


@dataclass(frozen=True)
class TopicDocuments:
    topic_id: str
    top: list[RankedDocument]
    relevant_top: list[RankedDocument]
    relevant_bottom: list[RankedDocument]


def extract_documents(
    selection: TopicSelection,
    rankings: Mapping[int, Ranking],
    qrels: Qrels,
    top_count: int,
    relevant_count: int,
) -> dict[str, TopicDocuments]:
    """トピックごとに、優位な側の層の順位で上位と適合文書を並べる。

    反対側の層の順位は、各文書に並べて付ける。適合文書は先頭 `relevant_count` 件を上位に、
    残りの末尾 `relevant_count` 件までを下位に分ける。重ねると、下位が取り落とした適合文書を
    表さなくなる。
    """
    document_ids = _shared_document_ids(rankings)
    position_of = {document_id: position for position, document_id in enumerate(document_ids)}

    groups = ((selection.shallow_favored, True), (selection.deep_favored, False))
    return {
        choice.topic_id: _extract(
            choice,
            favors_shallow,
            rankings,
            document_ids,
            position_of,
            qrels.get(choice.topic_id, {}),
            top_count,
            relevant_count,
        )
        for choices, favors_shallow in groups
        for choice in choices
    }


def _extract(
    choice: TopicChoice,
    favors_shallow: bool,
    rankings: Mapping[int, Ranking],
    document_ids: tuple[str, ...],
    position_of: Mapping[str, int],
    judgements: Mapping[str, int],
    top_count: int,
    relevant_count: int,
) -> TopicDocuments:
    shallow_ranking = rankings[_layer_of(choice.shallow)]
    deep_ranking = rankings[_layer_of(choice.deep)]
    shallow_rank = _ranks_of(shallow_ranking, choice.topic_id)
    deep_rank = _ranks_of(deep_ranking, choice.topic_id)
    favored_ranking, favored_rank = (
        (shallow_ranking, shallow_rank) if favors_shallow else (deep_ranking, deep_rank)
    )

    def ranked(position: int) -> RankedDocument:
        document_id = document_ids[position]
        return RankedDocument(
            document_id=document_id,
            shallow_rank=shallow_rank[position] + 1,
            deep_rank=deep_rank[position] + 1,
            grade=judgements.get(document_id),
        )

    row = favored_ranking.topic_ids.index(choice.topic_id)
    top = [ranked(position) for position in favored_ranking.order[row, :top_count].tolist()]

    relevant = sorted(
        (
            position_of[document_id]
            for document_id, grade in judgements.items()
            if grade >= RELEVANT_FROM and document_id in position_of
        ),
        key=lambda position: favored_rank[position],
    )
    return TopicDocuments(
        topic_id=choice.topic_id,
        top=top,
        relevant_top=[ranked(position) for position in relevant[:relevant_count]],
        relevant_bottom=[
            ranked(position) for position in relevant[relevant_count:][-relevant_count:]
        ],
    )


def _shared_document_ids(rankings: Mapping[int, Ranking]) -> tuple[str, ...]:
    """文書の並びは全層で同じ前提で、位置を使い回す。食い違えば位置が別の文書を指す。"""
    orders = {ranking.document_ids for ranking in rankings.values()}
    if len(orders) != 1:
        raise ValueError(
            "層によって文書の並びが異なります。同じキャッシュから検索し直した順位を渡してください。"
        )
    (document_ids,) = orders
    return document_ids


def _layer_of(result: TopicResult) -> int:
    (layer,) = result.configuration.layers
    return layer


def _ranks_of(ranking: Ranking, topic_id: str) -> list[int]:
    """文書の位置から順位 (0 始まり) を引く表。`order` は順位から文書への向きで、逆向きが要る。"""
    order = ranking.order[ranking.topic_ids.index(topic_id)]
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device)
    return ranks.tolist()
