"""書き出した結果を読み戻す。

配置は書き出し側が決める。ここではその配置に従ってファイルを読み、記録を結果の型に
戻す。
"""

from __future__ import annotations

import json
from pathlib import Path

from hidden_subspace.encoding.pooling import Pooling
from hidden_subspace.experiment.results import TopicResult
from hidden_subspace.experiment.writer import TOPIC_DIRECTORY, results_dir


def read_topic_results(
    root: Path, model_id: str, experiment: str, *, pooling: Pooling | None
) -> list[TopicResult]:
    """トピックごとの結果を読む。

    トピックのファイルは名前順に、1 ファイルの中は行の順に並べる。ファイルシステムが
    返す順に任せると、実行ごとに並びが揺れる。

    ディレクトリがあっても記録が 1 件もなければ拒否する。実験が途中で止まった跡で
    ある可能性が高く、空のまま返すと結果がないことに気づけない。
    """
    directory = results_dir(root, model_id, pooling, experiment) / TOPIC_DIRECTORY
    if not directory.is_dir():
        if pooling is None:
            command = "python evaluate_run.py で run を評価"
        else:
            command = (
                f"python run_experiment.py --collection {root.name} --experiment {experiment} "
                f"--model {model_id} --pooling {pooling} を実行"
            )
        raise FileNotFoundError(
            f"{experiment} の記録がありません: {directory}\n先に {command}してください。"
        )

    results = [
        TopicResult.from_record(json.loads(line))
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not results:
        raise ValueError(
            f"記録が 1 件もありません: {directory}\n実験が最後まで実行されたかを確認してください。"
        )
    return results
