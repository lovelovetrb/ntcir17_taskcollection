"""層・条件・類似度・トピックごとの指標と、層ごとの分散占有率を JSON Lines で書き、読み戻す。

本体の実験 (`hidden_subspace.experiment.writer`) と同じ形にする (ADR-0035)。

    output/<collection>/<model_id>/<pooling>/
      summary.jsonl          層・条件・類似度ごとに、全トピックを平均した指標
      topic/<topic_id>.jsonl 層・条件・類似度ごとの、そのトピックの指標
      variance.jsonl         層ごとの、除いた主成分の分散占有率

記録は拡張子が .jsonl なのでコミットされない (ADR-0024)。run.py を回せば作り直せる。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from hidden_subspace.experiment.writer import SUMMARY_FILE, TOPIC_DIRECTORY

RAW = "raw"
"""中心化も除去もしない条件。layer-sweep と同じベクトル。"""
VARIANCE_FILE = "variance.jsonl"


def removal_condition(count: int) -> str:
    """k = 0 は中心化だけをした条件で、`RAW` とは別。"""
    return f"k={count}"


@dataclass(frozen=True)
class TopicRecord:
    model_id: str
    layer: int
    condition: str
    similarity: str
    topic_id: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class SummaryRecord:
    model_id: str
    layer: int
    condition: str
    similarity: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class VarianceRecord:
    model_id: str
    layer: int
    count: int
    cumulative_share: float
    """上位 `count` 本の主成分の分散占有率の和。`count` が 0 なら 0。"""


def summarize(records: Sequence[TopicRecord]) -> list[SummaryRecord]:
    """層・条件・類似度ごとに、トピックの指標を平均する。並びは最初に現れた順。"""
    groups: dict[tuple[str, int, str, str], list[TopicRecord]] = defaultdict(list)
    for record in records:
        groups[(record.model_id, record.layer, record.condition, record.similarity)].append(record)
    return [
        SummaryRecord(
            model_id=model_id,
            layer=layer,
            condition=condition,
            similarity=similarity,
            metrics={
                name: sum(record.metrics[name] for record in group) / len(group)
                for name in group[0].metrics
            },
        )
        for (model_id, layer, condition, similarity), group in groups.items()
    ]


def write_results(
    directory: Path, records: Sequence[TopicRecord], variances: Sequence[VarianceRecord]
) -> None:
    """3 種類の記録を書く。回し直したときに古い行が残らないよう、ファイルは置き換える。"""
    by_topic: dict[str, list[TopicRecord]] = defaultdict(list)
    for record in records:
        by_topic[record.topic_id].append(record)
    for topic_id, topic_records in by_topic.items():
        _write_lines(topic_records, directory / TOPIC_DIRECTORY / f"{topic_id}.jsonl")
    _write_lines(summarize(records), directory / SUMMARY_FILE)
    _write_lines(variances, directory / VARIANCE_FILE)


def read_topic_records(directory: Path) -> list[TopicRecord]:
    """トピックのファイルは名前順に、1 ファイルの中は行の順に読む。"""
    topic_directory = directory / TOPIC_DIRECTORY
    _require(topic_directory)
    return [
        TopicRecord(**row)
        for path in sorted(topic_directory.glob("*.jsonl"))
        for row in _read_rows(path)
    ]


def read_variance_records(directory: Path) -> list[VarianceRecord]:
    _require(directory / VARIANCE_FILE)
    return [VarianceRecord(**row) for row in _read_rows(directory / VARIANCE_FILE)]


def _write_lines(
    records: Sequence[TopicRecord | SummaryRecord | VarianceRecord], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"記録がありません: {path}\n"
            "先に PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run "
            "--collection <コレクション> --model <モデル> --pooling <プーリング> "
            "を実行してください。"
        )


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
