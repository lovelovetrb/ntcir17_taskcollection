#!/usr/bin/env python3
"""run ファイルを層の実験と同じ定義で評価し、記録を run の隣に書き出す (ADR-0023, ADR-0025)。

run は `results/<collection>/<model_id>/<experiment>/` に置く (ADR-0026)。BM25 の run は
組織者のノートブックで (ADR-0022, `make bm25`)、TF ÷ トークン数 の run は `make bm25-tf` で
作る。ここでは run を読んで `evaluate` に通すだけで、コンテナも GPU も要らない。

使い方:
    python evaluate_run.py
    python evaluate_run.py --run results/ntcir1/tf-per-token/ntcir17-transfer/train/run.res.gz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hidden_subspace.collection import COLLECTION_NAMES, collection_paths, read_collection
from hidden_subspace.evaluation import CUTOFFS, align_relevance, evaluate
from hidden_subspace.experiment.results import (
    Configuration,
    DimensionChoice,
    results_of_evaluation,
)
from hidden_subspace.experiment.runner import ExperimentResults
from hidden_subspace.experiment.writer import write_results
from hidden_subspace.trec_run import ranking_from_run, read_run, relevance_of_retrieved

REPORTED = ("nDCG@10", "nDCG@1000", "Recall@1000")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("results/ntcir1/bm25/ntcir17-transfer/train/MyRun-BM25.res.gz"),
        help="評価する run ファイル。results/<collection>/<model_id>/<experiment>/ の下に置く",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    return parser.parse_args()


def locate(run_path: Path, results_root: Path) -> tuple[str, str, str]:
    """run の置き場所から collection と model_id と experiment を導く。記録は run の隣に書かれる。

    `results/<collection>/<model_id>/<experiment>/<run>` の先頭のディレクトリが collection、
    次が model_id、run のあるディレクトリまでの残りが experiment。
    """
    try:
        relative = run_path.parent.relative_to(results_root)
    except ValueError:
        raise SystemExit(
            f"run が {results_root}/ の下にありません: {run_path}\n"
            f"run を {results_root}/<collection>/<model_id>/<experiment>/ に置くか、"
            "--results で置き場所を指定してください。"
        ) from None
    if len(relative.parts) < 3:
        raise SystemExit(
            f"run の置き場所から collection と model_id と experiment を導けません: {run_path}\n"
            f"{results_root}/<collection>/<model_id>/<experiment>/ の下に置いてください。"
        )
    collection = relative.parts[0]
    if collection not in COLLECTION_NAMES:
        raise SystemExit(
            f"run の置き場所の {collection} はコレクションの名前ではありません: {run_path}\n"
            f"{results_root}/{{{','.join(COLLECTION_NAMES)}}}/ の下に置いてください。"
        )
    return collection, relative.parts[1], Path(*relative.parts[2:]).as_posix()


def main() -> int:
    arguments = parse_arguments()
    collection_name, model_id, experiment = locate(arguments.run, arguments.results)
    print(
        f"run {arguments.run} / collection {collection_name} / model_id {model_id} / "
        f"experiment {experiment}"
    )

    collection = read_collection(collection_paths(collection_name, arguments.data))
    topic_ids = [topic.topic_id for topic in collection.topics]
    document_ids = [document.document_id for document in collection.documents]
    print(f"文書 {len(document_ids):,} / トピック {len(topic_ids)}")

    run = read_run(arguments.run)
    missing = [topic_id for topic_id in topic_ids if topic_id not in run]
    short = sum(1 for topic_id in topic_ids if 0 < len(run.get(topic_id, [])) < 1000)
    print(f"run にあるトピック {len(run)} / 無いトピック {len(missing)} / 1000 件未満 {short}")

    ranking = ranking_from_run(run, topic_ids, document_ids)
    relevance = align_relevance(collection.qrels, topic_ids, document_ids)
    evaluation = evaluate(ranking, relevance_of_retrieved(relevance, run), CUTOFFS)

    # 層の記録と同じ型で書く。layers 以下は run の評価では意味を持たない (ADR-0023)
    configuration = Configuration(
        model_id=model_id,
        layers=(),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
    )
    summary, per_topic = results_of_evaluation(experiment, configuration, evaluation)
    written = write_results(
        arguments.results / collection_name, ExperimentResults([summary], per_topic)
    )

    print(f"\n書き出し {written}")
    for name in REPORTED:
        print(f"  {name:>12} {summary.metrics[name]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
