"""層・条件・トピックごとの指標と、層ごとの分散占有率を JSON Lines で書き、読み戻す。

記録は output/ に置くが、拡張子が .jsonl なのでコミットされない (ADR-0024)。
run.py を回せば作り直せる。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

RAW = "raw"
"""中心化も除去もしない条件。layer-sweep と同じベクトル。"""


def removal_condition(count: int) -> str:
    """k = 0 は中心化だけをした条件で、`RAW` とは別。"""
    return f"k={count}"


@dataclass(frozen=True)
class TopicRecord:
    model_id: str
    layer: int
    condition: str
    topic_id: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class VarianceRecord:
    model_id: str
    layer: int
    count: int
    cumulative_share: float
    """上位 `count` 本の主成分の分散占有率の和。`count` が 0 なら 0。"""


def write_records(records: Sequence[TopicRecord | VarianceRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[TopicRecord]:
    return [TopicRecord(**row) for row in _read_rows(path)]


def read_variance_records(path: Path) -> list[VarianceRecord]:
    return [VarianceRecord(**row) for row in _read_rows(path)]


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"記録がありません: {path}\n"
            "先に PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run を実行してください。"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
