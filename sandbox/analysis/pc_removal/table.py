"""run.py の要約 (summary.jsonl) から、層ごと・条件ごとの指標を Markdown の表にして標準出力に書く。

行は層、列は条件 (生、k=0 (中心化だけ)、k=1 …)。類似度 (内積・コサイン) ごとに 1 つずつ表を書く。
指標は `--metric` で選ぶ。GPU は使わない。

    PYTHONPATH=sandbox uv run python -m analysis.pc_removal.table \
        --collection ntcir1 --model cl-tohoku/bert-base-japanese-v3 --pooling mean
"""

from __future__ import annotations

import argparse

from analysis.pc_removal.records import RAW, SummaryRecord, read_summary_records, removal_condition
from analysis.pc_removal.run import COLLECTIONS, REMOVED_COUNTS, output_dir_of
from hidden_subspace.encoding.pooling import MEAN, POOLINGS
from hidden_subspace.vectors import COSINE, INNER_PRODUCT, Similarity

SIMILARITY_LABELS: dict[Similarity, str] = {INNER_PRODUCT: "内積", COSINE: "コサイン"}
CONDITIONS = (RAW, *(removal_condition(count) for count in REMOVED_COUNTS))
CONDITION_LABELS = {RAW: "生"}


def markdown_table(records: list[SummaryRecord], *, similarity: Similarity, metric: str) -> str:
    values = {
        (record.layer, record.condition): record.metrics[metric]
        for record in records
        if record.similarity == similarity
    }
    header = ["層", *(CONDITION_LABELS.get(condition, condition) for condition in CONDITIONS)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for layer in sorted({layer for layer, _ in values}):
        cells = [str(layer), *(f"{values[(layer, condition)]:.4f}" for condition in CONDITIONS)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    parser.add_argument("--model", required=True, help="モデルの識別子")
    parser.add_argument("--pooling", default=MEAN, choices=POOLINGS)
    parser.add_argument("--metric", default="nDCG@1000")
    arguments = parser.parse_args()

    records = read_summary_records(
        output_dir_of(arguments.collection, arguments.model, arguments.pooling)
    )
    tables = [
        f"{label} {arguments.metric}\n\n"
        + markdown_table(records, similarity=similarity, metric=arguments.metric)
        for similarity, label in SIMILARITY_LABELS.items()
    ]
    print("\n\n".join(tables))


if __name__ == "__main__":
    main()
