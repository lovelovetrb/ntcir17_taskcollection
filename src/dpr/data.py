"""mMARCO の三つ組をクエリごとにまとめ、Tevatron の学習データにする (ADR-0036)。

三つ組 (クエリ ID・適合文書の ID・適合しない文書の ID) をクエリごとにまとめ、ID を日本語の
クエリ文と本文に置き換える。1 クエリを 1 行にした jsonl で、Tevatron の
`positive_passages`・`negative_passages` の形にする。

- 負例は、三つ組でそのクエリと組になっている適合しない文書すべて (重複は除く)
- 適合文書が複数あるクエリは、すべてを `positive_passages` に入れる
- 負例が 7 件未満のクエリも除かない。Tevatron が重複を許して選ぶ
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dpr.download import COLLECTION, QUERIES, TRIPLES

TRAINING_FILE = Path("cache/dpr/data/train.jsonl")


@dataclass
class QueryPairs:
    """1 つのクエリと組になった文書の ID。並びは三つ組に最初に現れた順。"""

    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)


def read_texts(lines: Iterable[str]) -> dict[str, str]:
    """`ID<TAB>文` の行を読む。"""
    texts: dict[str, str] = {}
    for line in lines:
        item_id, text = line.rstrip("\n").split("\t", 1)
        texts[item_id] = text
    return texts


def group_triples(lines: Iterable[str]) -> dict[str, QueryPairs]:
    """三つ組をクエリごとにまとめる。同じ文書が何度現れても 1 件と数える。"""
    groups: dict[str, QueryPairs] = {}
    seen: dict[str, tuple[set[str], set[str]]] = {}
    for line in lines:
        query_id, positive_id, negative_id = line.rstrip("\n").split("\t")
        if query_id not in groups:
            groups[query_id] = QueryPairs()
            seen[query_id] = (set(), set())
        pairs = groups[query_id]
        seen_positives, seen_negatives = seen[query_id]
        if positive_id not in seen_positives:
            seen_positives.add(positive_id)
            pairs.positives.append(positive_id)
        if negative_id not in seen_negatives:
            seen_negatives.add(negative_id)
            pairs.negatives.append(negative_id)
    return groups


def check_ids(
    groups: dict[str, QueryPairs], queries: dict[str, str], passages: dict[str, str]
) -> None:
    """三つ組の ID が、日本語のクエリと文書集合にすべてあるかを確かめる。

    欠けたまま進めると、学習データから文が黙って抜ける。
    """
    missing_queries = [query_id for query_id in groups if query_id not in queries]
    missing_passages = {
        passage_id
        for pairs in groups.values()
        for passage_id in (*pairs.positives, *pairs.negatives)
        if passage_id not in passages
    }
    if missing_queries or missing_passages:
        raise ValueError(
            f"三つ組の ID が日本語のクエリ ({len(missing_queries)} 件) と文書集合 "
            f"({len(missing_passages)} 件) にありません。例: {missing_queries[:3]}、"
            f"{sorted(missing_passages)[:3]}。取ったファイルが正しいかを確かめてください。"
        )


def training_rows(
    groups: dict[str, QueryPairs], queries: dict[str, str], passages: dict[str, str]
) -> Iterator[dict[str, Any]]:
    """1 クエリを 1 行にする。並びは三つ組にクエリが最初に現れた順。"""
    for query_id, pairs in groups.items():
        yield {
            "query_id": query_id,
            "query": queries[query_id],
            "positive_passages": [
                {"docid": passage_id, "text": passages[passage_id]}
                for passage_id in pairs.positives
            ],
            "negative_passages": [
                {"docid": passage_id, "text": passages[passage_id]}
                for passage_id in pairs.negatives
            ],
        }


def write_rows(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """書いた行数を返す。

    途中で止まると半端なファイルが残り、次の実行がそれを完成したものとして読む。一時ファイルに
    書いてから名前を変える。一時ファイルの名前は呼び出しごとに変え、CLS と mean の学習を同時に
    始めても互いの一時ファイルを奪わないようにする。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".partial",
        delete=False,
    ) as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    Path(file.name).replace(path)
    return count


def prepare_training_data(mmarco_dir: Path, path: Path = TRAINING_FILE) -> Path:
    """学習データがあればそのまま使い、なければ取ったファイルから作る。"""
    if path.is_file():
        return path
    for name in (TRIPLES, QUERIES, COLLECTION):
        if not (mmarco_dir / name).is_file():
            raise FileNotFoundError(
                f"{mmarco_dir / name} がありません。"
                "先に uv run python -m dpr.download を実行してください。"
            )
    with (mmarco_dir / TRIPLES).open(encoding="utf-8") as file:
        groups = group_triples(file)
    with (mmarco_dir / QUERIES).open(encoding="utf-8") as file:
        queries = read_texts(file)
    with (mmarco_dir / COLLECTION).open(encoding="utf-8") as file:
        passages = read_texts(file)
    check_ids(groups, queries, passages)
    count = write_rows(training_rows(groups, queries, passages), path)
    print(f"学習データ {count:,} クエリを {path} に書きました。")
    return path
