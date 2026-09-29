"""LLM の判定 1 件の型と、JSON Lines での読み書き (issue #46)。

置き場所は `sandbox/results/word_sense/<model>/<topic_id>/<side>.jsonl` で、1 行が 1 文書の判定。
side (shallow / deep) はファイル名だけが持ち、記録には入れない。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CATEGORIES = ("一致", "一部一致", "別の意味で解釈", "無関係")
SIDES = ("shallow", "deep")


@dataclass(frozen=True)
class LlmOutput:
    model: str
    """判定に使ったモデルの名前とバージョン。"""
    category: str
    reason: str
    """「別の意味で解釈」のときだけ、クエリでの意味と検索テキストでの意味。それ以外は空。"""
    raw: str
    """応答の生の文字列。構造化出力の解析に失敗したときに追うため。"""


@dataclass(frozen=True)
class Judgement:
    topic_id: str
    title: str
    description: str
    model: str
    """検索に使った言語モデルの短い名前 (bert / simcse)。"""
    layer: int
    rank: int
    """1 始まり。"""
    document_id: str
    document_text: str
    llm: LlmOutput

    def as_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Judgement:
        fields = dict(record)
        fields["llm"] = LlmOutput(**fields["llm"])
        return cls(**fields)


def judgement_path(root: Path, model: str, topic_id: str, side: str) -> Path:
    if side not in SIDES:
        raise ValueError(f"side は {SIDES} のいずれかにしてください: {side}")
    return root / model / topic_id / f"{side}.jsonl"


def write_judgements(path: Path, judgements: Iterable[Judgement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for judgement in judgements:
            handle.write(json.dumps(judgement.as_record(), ensure_ascii=False) + "\n")


def read_judgements(path: Path) -> list[Judgement]:
    return [
        Judgement.from_record(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
