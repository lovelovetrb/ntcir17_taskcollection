"""書き出した結果を読み戻す。

配置は書き出し側が決める。ここではその配置に従ってファイルを読み、記録を結果の型に
戻す。
"""

from __future__ import annotations

import json
from pathlib import Path

from hidden_subspace.experiment.results import TopicResult
from hidden_subspace.experiment.writer import TOPIC_DIRECTORY


def read_topic_results(root: Path, model_id: str, experiment: str) -> list[TopicResult]:
    """トピックごとの結果を読む。

    トピックのファイルは名前順に、1 ファイルの中は行の順に並べる。ファイルシステムが
    返す順に任せると、実行ごとに並びが揺れる。

    ディレクトリがあっても記録が 1 件もなければ拒否する。実験が途中で止まった跡で
    ある可能性が高く、空のまま返すと結果がないことに気づけない。
    """
    directory = root / model_id / experiment / TOPIC_DIRECTORY
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{experiment} の記録がない: {directory}\n"
            f"先に python run_experiment.py --experiment {experiment} --model {model_id}"
        )

    results = [
        TopicResult.from_record(json.loads(line))
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not results:
        raise ValueError(f"記録が空: {directory}")
    return results
