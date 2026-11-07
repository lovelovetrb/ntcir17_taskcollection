"""NTCIR-2 の全文書で、品詞の大分類ごとに語の出現回数と、その品詞の語を含む文書の数を数える。

語は `pca.analysis.common.words` と同じく、512 トークンで打ち切った後の文章を MeCab
(unidic-lite) で分け、記号・補助記号を除いて数える (ADR-0031)。表を
`output/ntcir2.md` に書く。

    PYTHONPATH=sandbox uv run python -m analysis.pos_frequency \
        --model cl-tohoku/bert-base-japanese-v3
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from transformers import BertJapaneseTokenizer

from hidden_subspace.collection import collection_paths, read_collection
from hidden_subspace.corpus.text import document_text
from pca.analysis.common.words import Tagger, load_tokenizer, new_tagger, segment, truncate

COLLECTION = "ntcir2"
OUTPUT_PATH = Path(__file__).parent / "output" / f"{COLLECTION}.md"

_tokenizer: BertJapaneseTokenizer | None = None
_tagger: Tagger | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="打ち切りに使うトークナイザーのモデル")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    parser.add_argument("--processes", type=int, default=32)
    return parser.parse_args()


def _pos_counts(model_id: str, text: str) -> Counter[str]:
    # Pool の initializer で読み込むと、失敗したときに Pool がワーカーを作り直し続けて
    # 終わらない。最初のタスクで読み込み、失敗を親プロセスに返す
    global _tokenizer, _tagger
    if _tokenizer is None or _tagger is None:
        _tokenizer = load_tokenizer(model_id)
        _tagger = new_tagger()
    return Counter(word.pos for word in segment(truncate(text, _tokenizer, _tagger), _tagger))


def write_table(model_id: str, counts: Counter[str], documents: Counter[str], total: int) -> None:
    words = sum(counts.values())
    lines = [
        f"# 品詞の大分類ごとの出現回数と含む文書の数 ({COLLECTION})",
        "",
        f"文書 {total:,} 件、語 {words:,} 個 (記号・補助記号を除く)。"
        f"打ち切りは `{model_id}` のトークナイザーで行った。",
        "",
        "| 品詞 | 出現回数 | 全語に占める割合 | 含む文書 | 含む文書の割合 |",
        "|---|---|---|---|---|",
    ]
    for pos, n in counts.most_common():
        cells = (
            pos,
            f"{n:,}",
            f"{n / words:.2%}",
            f"{documents[pos]:,}",
            f"{documents[pos] / total:.1%}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    collection = read_collection(collection_paths(COLLECTION, arguments.data))
    texts = [document_text(document) for document in collection.documents]
    counts: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    with Pool(arguments.processes) as pool:
        for document_counts in pool.imap(
            partial(_pos_counts, arguments.model), texts, chunksize=500
        ):
            counts.update(document_counts)
            documents.update(document_counts.keys())
    write_table(arguments.model, counts, documents, len(texts))
    print(f"文書 {len(texts):,} 件の表を {OUTPUT_PATH} に書きました。")


if __name__ == "__main__":
    main()
