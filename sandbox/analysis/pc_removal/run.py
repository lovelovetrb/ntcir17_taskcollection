"""層ごとに文書集合の上位主成分を除いて検索し、トピック別の指標を記録する (issue #47)。

層ごとに、生 (中心化も除去もしない) と、文書の平均を引いて上位 k 本の主成分を除いた
条件を評価する。k = 0 は中心化だけをした条件。主成分は全文書から推定し、クエリにも
同じ平均と主成分を当てる。順位付けと評価は本体の実験と同じ関数を使う。

    CUDA_VISIBLE_DEVICES=<番号> PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch

from analysis.pc_removal.records import RAW, TopicRecord, removal_condition, write_records
from analysis.pc_removal.removal import estimate_common_components
from hidden_subspace.collection import Ntcir1Paths, read_collection
from hidden_subspace.encoding.cache import load_cache
from hidden_subspace.evaluation import CUTOFFS, Relevance, align_relevance, evaluate
from hidden_subspace.retrieval import rank_documents
from hidden_subspace.states import HiddenStates
from hidden_subspace.vectors import build_search_vectors

CACHE_ROOT = Path("cache")
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS = {"bert": "cl-tohoku/bert-base-japanese-v3"}
REMOVED_COUNTS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_layer(
    model_id: str,
    documents: HiddenStates,
    queries: HiddenStates,
    relevance: Relevance,
) -> list[TopicRecord]:
    """1 層ぶんの全条件を評価する。`documents` と `queries` はその層だけに絞ってある。"""
    (layer,) = documents.layer_indices

    def records_of(
        condition: str, document_states: HiddenStates, query_states: HiddenStates
    ) -> list[TopicRecord]:
        ranking = rank_documents(
            build_search_vectors(query_states, normalize_layers=False),
            build_search_vectors(document_states, normalize_layers=False),
        )
        scores = evaluate(ranking, relevance, CUTOFFS).scores
        return [
            TopicRecord(
                model_id=model_id,
                layer=layer,
                condition=condition,
                topic_id=topic_id,
                metrics={name: float(values[row].item()) for name, values in scores.items()},
            )
            for row, topic_id in enumerate(ranking.topic_ids)
        ]

    records = records_of(RAW, documents, queries)
    components = estimate_common_components(documents.values[:, 0, :], max(REMOVED_COUNTS))
    for count in REMOVED_COUNTS:
        removed_documents = components.remove(documents.values[:, 0, :], count).unsqueeze(1)
        removed_queries = components.remove(queries.values[:, 0, :], count).unsqueeze(1)
        records += records_of(
            removal_condition(count),
            replace(documents, values=removed_documents),
            replace(queries, values=removed_queries),
        )
    return records


def main() -> None:
    arguments = parse_arguments()
    collection = read_collection(Ntcir1Paths(arguments.data / "NTCIR-1"))

    for model, model_id in MODELS.items():
        documents = load_cache(CACHE_ROOT / model_id / "documents.pt")
        queries = load_cache(CACHE_ROOT / model_id / "queries.pt")
        relevance = align_relevance(collection.qrels, queries.item_ids, documents.item_ids).to(
            arguments.device
        )

        records: list[TopicRecord] = []
        for layer in documents.layer_indices:
            print(f"{model} layer {layer}")
            records += evaluate_layer(
                model_id,
                documents.select_layers([layer]).to(arguments.device),
                queries.select_layers([layer]).to(arguments.device),
                relevance,
            )
        write_records(records, OUTPUT_DIR / f"metrics-{model}.jsonl")


if __name__ == "__main__":
    main()
