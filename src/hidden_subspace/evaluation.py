"""順位を指標に変換する。

nDCG / Precision / Recall を指定した k で計算する (ADR-0013)。適合の判定は
2 値で、grade が 1 以上を適合とする (ADR-0011)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from hidden_subspace.retrieval import Ranking

Qrels = Mapping[str, Mapping[str, int]]
"""トピック ID から、文書 ID と grade の対応への写像。"""

RELEVANT_FROM = 1
"""この grade 以上を適合とする。"""


@dataclass(frozen=True)
class Relevance:
    """文書の並びに合わせた適合度。

    `values` は `[トピック数, 文書数]` で、位置 j の値は `document_ids[j]` が
    適合なら 1、そうでなければ 0。判定のない文書は 0 とする。

    `relevant_counts` は qrels に基づく適合文書の総数であり、コーパスに存在
    しない文書も含む。検索できない適合文書も Recall の分母に入れるため、
    `values` の合計とは一致しないことがある。
    """

    topic_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    values: Tensor
    relevant_counts: Tensor

    def to(self, device: str | torch.device) -> Relevance:
        """テンソルを指定した置き場所へ移す。"""
        return replace(
            self,
            values=self.values.to(device),
            relevant_counts=self.relevant_counts.to(device),
        )


@dataclass(frozen=True)
class Evaluation:
    """指標の計算結果。トピックごとの値を残す。"""

    topic_ids: tuple[str, ...]
    scores: dict[str, Tensor]

    def macro_average(self) -> dict[str, float]:
        """トピックごとの値を平均する。"""
        return {name: float(values.mean().item()) for name, values in self.scores.items()}


def align_relevance(
    qrels: Qrels, topic_ids: Sequence[str], document_ids: Sequence[str]
) -> Relevance:
    """qrels を文書の並びに合わせる。

    文書の並びは実験を通じて変わらないため、この対応付けは一度だけ行い、構成ごとの
    指標計算では使い回す。
    """
    position_of = {document_id: position for position, document_id in enumerate(document_ids)}
    values = torch.zeros(len(topic_ids), len(document_ids), dtype=torch.float32)
    counts = torch.zeros(len(topic_ids), dtype=torch.float32)

    for row, topic_id in enumerate(topic_ids):
        judgements = qrels.get(topic_id, {})
        for document_id, grade in judgements.items():
            if grade < RELEVANT_FROM:
                continue
            counts[row] += 1
            position = position_of.get(document_id)
            if position is not None:
                values[row, position] = 1.0

    return Relevance(
        topic_ids=tuple(topic_ids),
        document_ids=tuple(document_ids),
        values=values,
        relevant_counts=counts,
    )


def evaluate(ranking: Ranking, relevance: Relevance, ks: Sequence[int]) -> Evaluation:
    """順位と適合度から nDCG / Precision / Recall を計算する。"""
    if ranking.document_ids != relevance.document_ids:
        raise ValueError("順位と適合度が異なる文書の並びに基づいている")
    if ranking.topic_ids != relevance.topic_ids:
        raise ValueError("順位と適合度が異なるトピックに基づいている")

    in_rank_order = relevance.values.gather(1, ranking.order)
    document_count = in_rank_order.shape[1]
    device = in_rank_order.device
    positions = torch.arange(document_count, dtype=torch.float32, device=device)
    discount = 1.0 / torch.log2(positions + 2.0)

    scores: dict[str, Tensor] = {}
    for k in ks:
        cutoff = min(k, document_count)
        found = in_rank_order[:, :cutoff].sum(dim=1)
        scores[f"Precision@{k}"] = found / cutoff
        scores[f"Recall@{k}"] = _divide(found, relevance.relevant_counts)
        scores[f"nDCG@{k}"] = _divide(
            (in_rank_order[:, :cutoff] * discount[:cutoff]).sum(dim=1),
            _ideal_gain(relevance.relevant_counts, discount, cutoff),
        )

    return Evaluation(topic_ids=ranking.topic_ids, scores=scores)


def _ideal_gain(relevant_counts: Tensor, discount: Tensor, cutoff: int) -> Tensor:
    """適合文書を上位に並べたときの割引累積利得。"""
    reachable = torch.clamp(relevant_counts, max=cutoff)
    zero = torch.zeros(1, dtype=discount.dtype, device=discount.device)
    cumulative = torch.cat([zero, discount[:cutoff].cumsum(dim=0)])
    return cumulative[reachable.long()]


def _divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """分母が 0 のときは 0 とする。適合文書のないトピックで 0 除算にしないため。"""
    return torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)
