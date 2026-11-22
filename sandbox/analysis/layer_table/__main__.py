"""layer-sweep の記録から、層ごとの指標を Markdown の表にして標準出力に書く。

1 回に 1 つのモデルとプーリングを選ぶ。行は層、列は類似度 (内積・コサイン) ごとの指標。

    PYTHONPATH=sandbox uv run python -m analysis.layer_table --collection ntcir1 \
        --model dpr/FacebookAI/xlm-roberta-large/<run_name> --pooling cls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hidden_subspace.collection import COLLECTION_NAMES
from hidden_subspace.encoding.pooling import POOLINGS
from hidden_subspace.experiment.results import Configuration
from hidden_subspace.experiment.writer import SUMMARY_FILE, results_dir
from hidden_subspace.vectors import COSINE, INNER_PRODUCT, Similarity

EXPERIMENT = "layer-sweep"
METRICS = ("nDCG@10", "nDCG@1000", "Recall@1000")
SIMILARITY_LABELS: dict[Similarity, str] = {INNER_PRODUCT: "内積", COSINE: "コサイン"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTION_NAMES)
    parser.add_argument("--model", required=True, help="モデルの識別子")
    parser.add_argument("--pooling", required=True, choices=POOLINGS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    return parser.parse_args()


def read_layer_metrics(path: Path) -> dict[tuple[int, Similarity | None], dict[str, float]]:
    """(層, 類似度) ごとの全トピック平均の指標。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"{EXPERIMENT} の記録がありません: {path}\n"
            "先に python run_experiment.py --experiment layer-sweep を実行してください。"
        )
    table: dict[tuple[int, Similarity | None], dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        configuration = Configuration.from_record(record)
        (layer,) = configuration.layers
        table[(layer, configuration.similarity)] = record["metrics"]
    return table


def markdown_table(table: dict[tuple[int, Similarity | None], dict[str, float]]) -> str:
    columns = [(similarity, metric) for similarity in SIMILARITY_LABELS for metric in METRICS]
    header = ["層", *(f"{SIMILARITY_LABELS[s]} {m}" for s, m in columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for layer in sorted({layer for layer, _ in table}):
        cells = [str(layer), *(f"{table[(layer, s)][m]:.4f}" for s, m in columns)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    arguments = parse_arguments()
    root = arguments.results / arguments.collection
    path = results_dir(root, arguments.model, arguments.pooling, EXPERIMENT) / SUMMARY_FILE
    print(markdown_table(read_layer_metrics(path)))


if __name__ == "__main__":
    main()
