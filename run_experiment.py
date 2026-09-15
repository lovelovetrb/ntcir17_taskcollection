#!/usr/bin/env python3
"""指定した実験を回し、結果を書き出す。

エンコード済みのキャッシュがなければ作る。全 332,918 件のエンコードには 20 分
以上かかるが、2 回目以降は保存したものを読む。

使い方:
    python run_experiment.py --experiment layer-sweep
    python run_experiment.py --experiment layer-sweep --model cl-nagoya/unsup-simcse-ja-base
"""

from __future__ import annotations

import argparse
import os
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import torch

from hidden_subspace.collection import Collection, Ntcir1Paths, read_collection
from hidden_subspace.corpus.text import document_text, query_text
from hidden_subspace.encoding.cache import encode_or_load
from hidden_subspace.encoding.encoder import Encoder
from hidden_subspace.evaluation import CUTOFFS, Relevance, align_relevance
from hidden_subspace.experiment.plans import build_plans, plan_names
from hidden_subspace.experiment.results import Configuration, ConfigurationResult
from hidden_subspace.experiment.runner import run_experiment
from hidden_subspace.experiment.writer import write_results
from hidden_subspace.states import HiddenStates

DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"
REPORTED = ("nDCG@10", "nDCG@1000", "Recall@1000")
PRIMARY = "nDCG@1000"
LABEL_WIDTH = 44
FULL_REPORT_LIMIT = 50
"""これを超える構成数のときは、主指標の上位だけを表にする。"""
TOP_REPORTED = 20


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=plan_names())
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace のモデル識別子")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def encoded(
    model_id: str, cache_root: Path, kind: str, texts: Sequence[str], ids: Sequence[str]
) -> HiddenStates:
    path = cache_root / model_id / f"{kind}.pt"

    def build() -> HiddenStates:
        print(f"  {kind}: {len(texts):,} 件をエンコード")
        started = time.perf_counter()
        states = Encoder(model_id).encode(texts, ids)
        print(f"  {kind}: {time.perf_counter() - started:.0f} 秒")
        return states

    return encode_or_load(path, build)


def prepare(
    collection: Collection, model_id: str, cache_root: Path, device: str
) -> tuple[HiddenStates, HiddenStates, Relevance]:
    documents = encoded(
        model_id,
        cache_root,
        "documents",
        [document_text(d) for d in collection.documents],
        [d.document_id for d in collection.documents],
    ).to(device)
    queries = encoded(
        model_id,
        cache_root,
        "queries",
        [query_text(t) for t in collection.topics],
        [t.topic_id for t in collection.topics],
    ).to(device)
    relevance = align_relevance(collection.qrels, queries.item_ids, documents.item_ids).to(device)
    return documents, queries, relevance


def configuration_label(configuration: Configuration) -> str:
    """構成を 1 行のラベルにする。

    層と次元の両方を書く。層だけを書くと、層を固定して次元を変える実験で
    すべての行が同じラベルになる。
    """
    layers = "-".join(str(layer) for layer in configuration.layers)
    choice = configuration.dimensions
    parameters = "".join(f" {name}={value}" for name, value in choice.parameters.items())
    return f"層 {layers} / 次元 {choice.kind}{parameters}"


def padded(text: str, width: int) -> str:
    """右に空白を詰める。全角は 2 桁と数える。

    書式指定の桁数は文字数で数えるため、全角を含むラベルはそのままでは桁が
    ずれる。
    """
    columns = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - columns, 0)


def report(results: Sequence[ConfigurationResult]) -> None:
    """構成ごとの指標を表にする。

    構成が多いときは主指標の上位だけを出す。8,000 を超える行を端末に流しても
    読めない。記録はファイルが正であり、この表は目視の便宜である (ADR-0019)。
    """
    shown = results
    print()
    if len(results) > FULL_REPORT_LIMIT:
        shown = sorted(results, key=lambda r: r.metrics[PRIMARY], reverse=True)[:TOP_REPORTED]
        print(f"{len(results):,} 構成のうち {PRIMARY} の上位 {len(shown)} 件")

    header = padded("構成", LABEL_WIDTH) + " " + " ".join(f"{name:>12}" for name in REPORTED)
    print(header)
    for result in shown:
        scores = " ".join(f"{result.metrics[name]:>12.4f}" for name in REPORTED)
        print(f"{padded(configuration_label(result.configuration), LABEL_WIDTH)} {scores}")

    for result in results:
        if result.zero_norm_queries or result.zero_norm_documents:
            print(
                f"向きを持たないベクトル: クエリ {result.zero_norm_queries} 件 / "
                f"文書 {result.zero_norm_documents} 件 "
                f"({configuration_label(result.configuration)})"
            )


def main() -> int:
    arguments = parse_arguments()
    print(f"実験 {arguments.experiment} / モデル {arguments.model}")

    collection = read_collection(Ntcir1Paths(arguments.data / "NTCIR-1"))
    print(
        f"文書 {len(collection.documents):,} / トピック {len(collection.topics)} / "
        f"判定 {sum(len(v) for v in collection.qrels.values()):,}"
    )

    documents, queries, relevance = prepare(
        collection, arguments.model, arguments.cache, arguments.device
    )

    plans = build_plans(arguments.experiment, documents.layer_indices, documents.dimension_indices)
    print(f"\n{len(plans)} 構成を評価 ({arguments.device})")
    started = time.perf_counter()
    results = run_experiment(
        experiment=arguments.experiment,
        model_id=arguments.model,
        documents=documents,
        queries=queries,
        relevance=relevance,
        plans=plans,
        ks=CUTOFFS,
    )
    print(f"  {time.perf_counter() - started:.1f} 秒")

    print(f"\n書き出し {write_results(arguments.results, results)}")
    report(results.configurations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
