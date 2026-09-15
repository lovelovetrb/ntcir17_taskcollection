"""検索し直した順位を、layer-sweep の記録と照合する。

順位は置き場所によって変わるため、表示する順位が記録された指標を出した順位と同じで
あることを、結果ファイルを作るたびに確かめる (ADR-0020)。
"""

from __future__ import annotations

from collections.abc import Mapping

from hidden_subspace.evaluation import CUTOFFS, Evaluation, Relevance, evaluate
from hidden_subspace.retrieval import Ranking
from layer_inspector.selection import TopicSelection


def verify_against_records(
    selection: TopicSelection,
    rankings: Mapping[int, Ranking],
    relevance: Relevance,
) -> None:
    """選んだトピックの記録ごとに指標を計算し直し、1 つでも食い違えば止める。

    許容誤差は設けない。誤差を許すと、スコアの近い文書が入れ替わった順位を見逃す。
    """
    evaluations = {
        layer: evaluate(ranking, relevance, CUTOFFS) for layer, ranking in rankings.items()
    }

    mismatches: list[str] = []
    for choice in (*selection.shallow_favored, *selection.deep_favored):
        for record in (choice.shallow, choice.deep):
            (layer,) = record.configuration.layers
            recomputed = _metrics_of(evaluations[layer], choice.topic_id)
            recorded = dict(record.metrics)
            if recomputed == recorded:
                continue
            differences = ", ".join(
                f"{name} 記録 {recorded.get(name)} / 再計算 {recomputed.get(name)}"
                for name in sorted(recorded.keys() | recomputed.keys())
                if recorded.get(name) != recomputed.get(name)
            )
            mismatches.append(f"トピック {choice.topic_id} 層 {layer}: {differences}")

    if mismatches:
        raise ValueError(
            "検索し直した順位から計算した指標が、記録と一致しません。\n"
            + "\n".join(mismatches)
            + "\nGPU で検索し直したか、記録とキャッシュが同じ実行のものかを確認してください。"
        )


def _metrics_of(evaluation: Evaluation, topic_id: str) -> dict[str, float]:
    row = evaluation.topic_ids.index(topic_id)
    return {name: float(values[row].item()) for name, values in evaluation.scores.items()}
