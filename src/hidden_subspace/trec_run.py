"""TREC 形式の run ファイルを、層の実験と同じ評価に渡せる形にする (ADR-0023)。

run には各トピックの上位 1,000 件しか無い。残りをコレクションの並び順で埋めて `Ranking` に
し、埋めた文書 (run が見つけていない文書) は適合度を 0 にして評価から外す。trec_eval が
run に無い文書を利得 0 に数えるのと同じ意味になる。
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import torch

from hidden_subspace.evaluation import Relevance
from hidden_subspace.retrieval import Ranking

Run = Mapping[str, Sequence[tuple[str, float]]]
"""トピック ID から、順位の高い順に並べた (文書 ID, スコア) への対応。"""


def read_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """1 行は空白区切りで (トピック, Q0, 文書, 順位, スコア, 名前)。

    行の並びは順位順とは限らないため、順位の列で並べ直す。`.gz` ならそのまま展開して読む。
    """
    rows: dict[str, list[tuple[int, str, float]]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            topic_id, _, document_id, rank, score = fields[:5]
            rows.setdefault(topic_id, []).append((int(rank), document_id, float(score)))
    return {
        topic_id: [(document_id, score) for _, document_id, score in sorted(entries)]
        for topic_id, entries in rows.items()
    }


def ranking_from_run(run: Run, topic_ids: Sequence[str], document_ids: Sequence[str]) -> Ranking:
    """run の文書を順位どおりに先頭へ置き、残りをコレクションの並び順で埋める。

    埋めた部分のスコアは、run のどの値よりも小さくする。run に無いトピックは全文書が
    埋めた部分になる。
    """
    position_of = {document_id: position for position, document_id in enumerate(document_ids)}
    count = len(document_ids)
    order = torch.empty(len(topic_ids), count, dtype=torch.long)
    scores = torch.empty(len(topic_ids), count, dtype=torch.float32)

    for row, topic_id in enumerate(topic_ids):
        entries = run.get(topic_id, [])
        unknown = [document_id for document_id, _ in entries if document_id not in position_of]
        if unknown:
            raise ValueError(
                f"トピック {topic_id} の run にある文書 {unknown[:3]} がコレクションにありません。"
                "同じコレクションで作った run を渡してください。"
            )
        head = torch.tensor([position_of[d] for d, _ in entries], dtype=torch.long)
        remaining = torch.ones(count, dtype=torch.bool)
        remaining[head] = False
        order[row] = torch.cat([head, torch.arange(count)[remaining]])

        head_scores = torch.tensor([score for _, score in entries], dtype=torch.float32)
        floor = float(head_scores.min()) - 1.0 if entries else 0.0
        scores[row] = torch.cat([head_scores, torch.full((count - len(entries),), floor)])

    return Ranking(
        topic_ids=tuple(topic_ids),
        document_ids=tuple(document_ids),
        order=order,
        scores=scores,
    )


def relevance_of_retrieved(relevance: Relevance, run: Run) -> Relevance:
    """run に無い文書の適合度を 0 にする。適合文書の総数は変えない。

    見つけていない適合文書は nDCG と Precision に入らず、理想の DCG と Recall の分母には
    残る。
    """
    position_of = {
        document_id: position for position, document_id in enumerate(relevance.document_ids)
    }
    retrieved = torch.zeros_like(relevance.values)
    for row, topic_id in enumerate(relevance.topic_ids):
        positions = [
            position_of[document_id]
            for document_id, _ in run.get(topic_id, [])
            if document_id in position_of
        ]
        retrieved[row, positions] = 1.0
    return replace(relevance, values=relevance.values * retrieved)
