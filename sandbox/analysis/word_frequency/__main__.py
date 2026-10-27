"""コレクションの全文書で、語ごとの出現回数と、語を含む文書・含まない文書の数を数える (ADR-0029)。

標準化平均差 δ を使う語と、文書中の割合の r² で見る語の境目 (含まない文書が何件以上あれば
δ を使うか) を決めるための集計。語は表層形・品詞の大分類・中分類の組で区別し、中分類がない
品詞の語は表層形と大分類の組で区別する。

output/<コレクション>/ に次の 3 つを書く。jsonl はコミットしない。

- `word-frequencies.jsonl`: 全語の出現回数と、含む文書・含まない文書の数。含む文書の多い順
- `word-counts.jsonl`: 全語の出現回数と、その順位。出現回数の多い順
- `summary.md`: 含まない文書の数ごとの語の数と、含む文書の割合が高い語の一覧

どちらの jsonl にも、語ごとに品詞の小分類とその出現回数 (`pos3`) を添える。

    PYTHONPATH=sandbox uv run python -m analysis.word_frequency --collection ntcir1
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from transformers import AutoTokenizer, BertJapaneseTokenizer

from analysis.word_frequency.words import (
    TaggedWord,
    Tagger,
    Word,
    new_tagger,
    segment,
    truncate,
)
from hidden_subspace.collection import collection_paths, read_collection
from hidden_subspace.corpus.text import document_text

MODEL_ID = "cl-tohoku/bert-base-japanese-v3"
OUTPUT_DIR = Path(__file__).parent / "output"
ABSENT_THRESHOLDS = (10, 100, 1_000, 10_000, 100_000)
COLLECTIONS = ("ntcir1", "ntcir2")
LISTED_SHARE = 0.9
"""含む文書の割合がこれ以上の語を summary.md に並べる。"""

_tokenizer: BertJapaneseTokenizer | None = None
_tagger: Tagger | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTIONS)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    parser.add_argument("--processes", type=int, default=32)
    return parser.parse_args()


def _start_worker() -> None:
    global _tokenizer, _tagger
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if not isinstance(tokenizer, BertJapaneseTokenizer):
        raise TypeError(f"{MODEL_ID} のトークナイザーが BertJapaneseTokenizer ではありません。")
    _tokenizer = tokenizer
    _tagger = new_tagger()


def _words_of(text: str) -> list[TaggedWord]:
    assert _tokenizer is not None and _tagger is not None
    return segment(truncate(text, _tokenizer), _tagger)


def pos3_of(tagged: Counter[TaggedWord]) -> dict[Word, list[dict[str, str | int]]]:
    """語ごとに、品詞の小分類を出現回数の多い順に並べる。"""
    pos3: dict[Word, list[dict[str, str | int]]] = {}
    for tagged_word, n in tagged.most_common():
        pos3.setdefault(tagged_word.word, []).append({"pos3": tagged_word.pos3, "count": n})
    return pos3


def pos_label(word: Word) -> str:
    return word.pos if word.pos2 is None else f"{word.pos}-{word.pos2}"


def write_summary(
    collection: str, counts: Counter[Word], documents: Counter[Word], total: int, path: Path
) -> None:
    lines = [
        f"# 語ごとの出現文書数 ({collection})",
        "",
        f"文書 {total:,} 件、語 (表層形と品詞の大分類・中分類の組) {len(documents):,} 種類。",
        "",
        "## 含まない文書の数ごとの語の数",
        "",
        "| 含まない文書の数 | 語の数 |",
        "|---|---|",
    ]
    absent = [total - n for n in documents.values()]
    for threshold in ABSENT_THRESHOLDS:
        lines.append(f"| {threshold:,} 未満 | {sum(a < threshold for a in absent):,} |")
    lines += [
        "",
        f"## 含む文書の割合が {LISTED_SHARE:.0%} 以上の語",
        "",
        "| 語 | 品詞 | 出現回数 | 含む文書 | 含まない文書 | 含む文書の割合 |",
        "|---|---|---|---|---|---|",
    ]
    for word, n in documents.most_common():
        if n / total < LISTED_SHARE:
            break
        cells = (
            word.surface,
            pos_label(word),
            f"{counts[word]:,}",
            f"{n:,}",
            f"{total - n:,}",
            f"{n / total:.2%}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    collection = read_collection(collection_paths(arguments.collection, arguments.data))
    texts = [document_text(document) for document in collection.documents]

    counts: Counter[Word] = Counter()
    documents: Counter[Word] = Counter()
    tagged: Counter[TaggedWord] = Counter()
    with Pool(arguments.processes, initializer=_start_worker) as pool:
        for tagged_words in pool.imap(_words_of, texts, chunksize=500):
            words = [tagged_word.word for tagged_word in tagged_words]
            counts.update(words)
            documents.update(set(words))
            tagged.update(tagged_words)
    pos3 = pos3_of(tagged)

    output_dir = OUTPUT_DIR / arguments.collection
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(texts)
    with (output_dir / "word-frequencies.jsonl").open("w", encoding="utf-8") as file:
        for word, n in documents.most_common():
            row = {
                "surface": word.surface,
                "pos": word.pos,
                "pos2": word.pos2,
                "count": counts[word],
                "documents": n,
                "absent": total - n,
                "pos3": pos3[word],
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "word-counts.jsonl").open("w", encoding="utf-8") as file:
        for rank, (word, n) in enumerate(counts.most_common(), start=1):
            row = {
                "word": word.surface,
                "pos": word.pos,
                "pos2": word.pos2,
                "count": n,
                "rank": rank,
                "pos3": pos3[word],
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(arguments.collection, counts, documents, total, output_dir / "summary.md")
    print(f"文書 {total:,} 件、語 {len(documents):,} 種類を {output_dir} に書きました。")


if __name__ == "__main__":
    main()
