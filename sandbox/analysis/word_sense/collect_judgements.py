"""浅い側・深い側の最良の層で検索し直し、上位の文書を Gemini に判定させて残す (issue #46)。

流れ: 層の記録を読む → トピックごとに使う層を決める → キャッシュの隠れ状態で検索する →
上位 100 件の文書とクエリ (TITLE と DESCRIPTION) を Gemini に渡す → JSON Lines に書く。

既に書かれているトピック・側は飛ばすので、途中で止まっても続きから回せる。

    PYTHONPATH=sandbox uv run python -m analysis.word_sense.collect_judgements --model bert
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from analysis.word_sense.judge import Judge
from analysis.word_sense.records import SIDES, Judgement, judgement_path, write_judgements
from analysis.word_sense.retrieval import (
    DEEP_LAYERS,
    SHALLOW_LAYERS,
    best_layers,
    rank_layers,
    top_documents,
)
from hidden_subspace.collection import Ntcir1Paths, read_collection
from hidden_subspace.corpus.text import document_text
from hidden_subspace.corpus.topics import Topic
from hidden_subspace.encoding.cache import load_cache
from hidden_subspace.experiment.reader import read_topic_results

MODELS = {
    "bert": "cl-tohoku/bert-base-japanese-v3",
    "simcse": "cl-nagoya/unsup-simcse-ja-base",
}
GEMINI_MODEL = "gemini-3.8-flash"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--count", type=int, default=100, help="各側で判定する上位の件数")
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    parser.add_argument("--workers", type=int, default=8, help="Gemini を並行して呼ぶ数")
    parser.add_argument("--topics", nargs="*", help="指定したトピックだけ回す (動作確認用)")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
    )
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("sandbox/results/word_sense"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def judge_side(
    executor: ThreadPoolExecutor,
    judge: Judge,
    *,
    topic: Topic,
    model: str,
    layer: int,
    ranked: Sequence[tuple[int, str]],
    texts: Mapping[str, str],
) -> list[Judgement]:
    """1 トピック・1 側の上位文書を並行して判定する。順位の順に返す。"""

    def judged(item: tuple[int, str]) -> Judgement:
        rank, document_id = item
        return Judgement(
            topic_id=topic.topic_id,
            title=topic.title,
            description=topic.description,
            model=model,
            layer=layer,
            rank=rank,
            document_id=document_id,
            document_text=texts[document_id],
            llm=judge.judge(topic.title, topic.description, texts[document_id]),
        )

    return list(executor.map(judged, ranked))


def tally(judgements: Sequence[Judgement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for judgement in judgements:
        counts[judgement.llm.category] = counts.get(judgement.llm.category, 0) + 1
    return counts


def main() -> int:
    arguments = parse_arguments()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY が設定されていません。.env を読み込んでから実行してください。"
        )
    model_id = MODELS[arguments.model]

    collection = read_collection(Ntcir1Paths(arguments.data / "NTCIR-1"))
    topics = {topic.topic_id: topic for topic in collection.topics}
    texts = {document.document_id: document_text(document) for document in collection.documents}
    layers = best_layers(read_topic_results(arguments.results, model_id, "layer-sweep"))
    wanted = arguments.topics or sorted(layers)

    documents = load_cache(arguments.cache / model_id / "documents.pt").to(arguments.device)
    queries = load_cache(arguments.cache / model_id / "queries.pt").to(arguments.device)
    rankings = rank_layers((*SHALLOW_LAYERS, *DEEP_LAYERS), documents, queries)
    print(f"検索 {len(wanted)} トピック、{len(SIDES)} 側ずつ ({arguments.device})")

    judge = Judge(arguments.gemini_model, api_key)
    print(f"判定 {judge.model_label} / 各側 {arguments.count} 件 / 並行 {arguments.workers}")

    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        for topic_id in wanted:
            for side in SIDES:
                path = judgement_path(arguments.out, arguments.model, topic_id, side)
                if path.is_file():
                    continue
                layer = layers[topic_id].layer_of(side)
                judgements = judge_side(
                    executor,
                    judge,
                    topic=topics[topic_id],
                    model=arguments.model,
                    layer=layer,
                    ranked=top_documents(rankings[layer], topic_id, arguments.count),
                    texts=texts,
                )
                write_judgements(path, judgements)
                title = topics[topic_id].title
                print(f"  {topic_id} {title} / {side} (層 {layer}): {tally(judgements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
