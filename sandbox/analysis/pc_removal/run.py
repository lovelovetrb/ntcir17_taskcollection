"""層ごとに文書集合の上位主成分を除いて検索し、トピック別の指標と分散占有率を記録する (issue #47)。

層ごとに、生 (中心化も除去もしない) と、文書の平均を引いて上位 k 本の主成分を除いた
条件を評価する。k = 0 は中心化だけをした条件。平均と主成分は全文書から推定し、クエリにも
同じ平均と主成分を当てる。順位付けと評価は本体の実験と同じ関数を使う。

`--collection ntcir2-unique` では、NTCIR-2 の文書のうち NTCIR-1 に含まれないものだけから
平均と主成分を推定する。除く処理と評価は NTCIR-2 の全文書に当てる。

キャッシュは `cache/<コレクション>/` から読み、記録は `output/<--collection の値>/` に書く。

    CUDA_VISIBLE_DEVICES=<番号> PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run \
        --collection ntcir1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from analysis.pc_removal.records import (
    RAW,
    TopicRecord,
    VarianceRecord,
    removal_condition,
    write_records,
)
from analysis.pc_removal.removal import estimate_common_components
from hidden_subspace.collection import Ntcir2Paths, collection_paths, read_collection
from hidden_subspace.corpus.documents import as_ntcir2_document_id, read_documents
from hidden_subspace.encoding.cache import load_cache
from hidden_subspace.evaluation import CUTOFFS, Relevance, align_relevance, evaluate
from hidden_subspace.retrieval import rank_documents
from hidden_subspace.states import HiddenStates
from hidden_subspace.vectors import build_search_vectors

CACHE_ROOT = Path("cache")
OUTPUT_DIR = Path(__file__).parent / "output"
DAPT_RUN_NAME = "bert-base-japanese-v3-ep3-bs128-lr5e-05-wu0.06"
MODELS = {
    "bert": "cl-tohoku/bert-base-japanese-v3",
    "simcse": "cl-nagoya/unsup-simcse-ja-base",
    "bert-dapt": f"dapt/cl-tohoku/bert-base-japanese-v3/{DAPT_RUN_NAME}",
}
REMOVED_COUNTS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200)
COLLECTIONS = {"ntcir1": "ntcir1", "ntcir2": "ntcir2", "ntcir2-unique": "ntcir2"}
"""`--collection` の値と、キャッシュと配布物を読むコレクション。"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def unique_document_mask(documents: HiddenStates, data: Path) -> Tensor:
    """NTCIR-2 の文書のうち、NTCIR-1 に含まれないものを True にする。

    NTCIR-1 の文書は、NTCIR-2 を読むときと同じ配布物から文書だけを読む。
    """
    paths = collection_paths("ntcir2", data)
    assert isinstance(paths, Ntcir2Paths)
    ntcir1_documents = read_documents(
        paths.ntcir1_root / paths.ntcir1_documents_archive, paths.ntcir1_documents_member
    )
    ntcir1_ids = {as_ntcir2_document_id(document.document_id) for document in ntcir1_documents}
    return torch.tensor([item_id not in ntcir1_ids for item_id in documents.item_ids])


def evaluate_layer(
    model_id: str,
    documents: HiddenStates,
    queries: HiddenStates,
    relevance: Relevance,
    estimation_mask: Tensor,
) -> tuple[list[TopicRecord], list[VarianceRecord]]:
    """1 層ぶんの全条件を評価し、除いた主成分の分散占有率も返す。

    `documents` と `queries` はその層だけに絞ってある。平均と主成分は `estimation_mask` が
    True の文書だけから推定する。
    """
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
    components = estimate_common_components(
        documents.values[estimation_mask, 0, :], max(REMOVED_COUNTS)
    )
    variances = [
        VarianceRecord(
            model_id=model_id,
            layer=layer,
            count=count,
            cumulative_share=float(components.variance_shares[:count].sum().item()),
        )
        for count in REMOVED_COUNTS
    ]
    for count in REMOVED_COUNTS:
        removed_documents = components.remove(documents.values[:, 0, :], count).unsqueeze(1)
        removed_queries = components.remove(queries.values[:, 0, :], count).unsqueeze(1)
        records += records_of(
            removal_condition(count),
            replace(documents, values=removed_documents),
            replace(queries, values=removed_queries),
        )
    return records, variances


def main() -> None:
    arguments = parse_arguments()
    collection_name = COLLECTIONS[arguments.collection]
    collection = read_collection(collection_paths(collection_name, arguments.data))
    cache_dir = CACHE_ROOT / collection_name
    output_dir = OUTPUT_DIR / arguments.collection

    for model, model_id in MODELS.items():
        documents = load_cache(cache_dir / model_id / "documents.pt")
        queries = load_cache(cache_dir / model_id / "queries.pt")
        relevance = align_relevance(collection.qrels, queries.item_ids, documents.item_ids).to(
            arguments.device
        )
        if arguments.collection == "ntcir2-unique":
            estimation_mask = unique_document_mask(documents, arguments.data)
        else:
            estimation_mask = torch.ones(len(documents.item_ids), dtype=torch.bool)
        estimation_mask = estimation_mask.to(arguments.device)
        print(f"推定に使う文書 {int(estimation_mask.sum()):,} / {len(documents.item_ids):,}")

        records: list[TopicRecord] = []
        variances: list[VarianceRecord] = []
        for layer in documents.layer_indices:
            print(f"{model} layer {layer}")
            layer_records, layer_variances = evaluate_layer(
                model_id,
                documents.select_layers([layer]).to(arguments.device),
                queries.select_layers([layer]).to(arguments.device),
                relevance,
                estimation_mask,
            )
            records += layer_records
            variances += layer_variances
        write_records(records, output_dir / f"metrics-{model}.jsonl")
        write_records(variances, output_dir / f"variance-{model}.jsonl")


if __name__ == "__main__":
    main()
