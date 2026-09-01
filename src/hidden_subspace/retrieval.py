"""検索ベクトルから順位を付ける。

順位は打ち切らない。上位何件を見るかは指標の側が決める。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hidden_subspace.vectors import SearchVectors


@dataclass(frozen=True)
class Ranking:
    """トピックごとに全文書を並べた結果。

    `order` と `scores` は `[トピック数, 文書数]` で、いずれも順位の高い順に並ぶ。
    `order[i][j]` は i 番目のトピックで j 位になった文書の `document_ids` への添字、
    `scores[i][j]` はそのときのスコア。文書 ID を全順位ぶん文字列として持つのを
    避けている。
    """

    topic_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    order: Tensor
    scores: Tensor

    def ranked_ids_for(self, topic_id: str) -> tuple[str, ...]:
        """指定したトピックについて、順位の高い順に文書 ID を返す。"""
        row = self.topic_ids.index(topic_id)
        return tuple(self.document_ids[i] for i in self.order[row].tolist())


def rank_documents(queries: SearchVectors, documents: SearchVectors) -> Ranking:
    """クエリごとに全文書を内積の降順で並べる。

    同点は文書の並び順で決める。順序が実行ごとに揺れると実験を再現できないため。
    すべての成分が 0 のベクトルは互いにスコア 0 で並ぶため、同点は実際に起こりうる。
    """
    if queries.layer_indices != documents.layer_indices:
        raise ValueError(
            "クエリと文書が異なる層から作られている: "
            f"{queries.layer_indices} と {documents.layer_indices}"
        )
    if queries.dimension_indices != documents.dimension_indices:
        raise ValueError(
            "クエリと文書が異なる次元から作られている: "
            f"{queries.dimension_indices} と {documents.dimension_indices}"
        )
    if queries.values.shape[1] != documents.values.shape[1]:
        raise ValueError(
            f"ベクトルの長さが違う: {queries.values.shape[1]} と {documents.values.shape[1]}"
        )

    scores = queries.values @ documents.values.T
    sorted_scores, order = torch.sort(scores, dim=1, descending=True, stable=True)
    return Ranking(
        topic_ids=queries.item_ids,
        document_ids=documents.item_ids,
        order=order,
        scores=sorted_scores,
    )
