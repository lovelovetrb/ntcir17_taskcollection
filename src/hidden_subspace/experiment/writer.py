"""結果を JSON Lines として書き出す。

配置は `(model_id)/(experiment)/` の下に、構成ごとの `summary.jsonl` と
トピックごとの `topic/(topic_id).jsonl`。トピックのファイル数は構成の数に
よらず一定になる。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hidden_subspace.experiment.runner import ExperimentResults

SUMMARY_FILE = "summary.jsonl"
TOPIC_DIRECTORY = "topic"


def write_results(root: Path, results: ExperimentResults) -> Path:
    """結果を書き出し、書き出し先のディレクトリを返す。

    構成ごとにファイルを開き直すと開閉が構成数だけ増えるため、まとめて書く。
    同じ実験を回し直したときに古い行が残らないよう、ファイルは置き換える。
    """
    if not results.configurations:
        raise ValueError("書き出す結果がない")

    first = results.configurations[0]
    base = root / first.configuration.model_id / first.experiment
    (base / TOPIC_DIRECTORY).mkdir(parents=True, exist_ok=True)

    _write_lines(base / SUMMARY_FILE, (r.as_record() for r in results.configurations))

    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic_result in results.topics:
        by_topic[topic_result.topic_id].append(topic_result.as_record())
    for topic_id, records in by_topic.items():
        _write_lines(base / TOPIC_DIRECTORY / f"{topic_id}.jsonl", records)

    return base


def _write_lines(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
