"""TF ÷ トークン数 の run を、BM25 と同じ Terrier のインデックスから作る (ADR-0025)。

コンテナの中で動く。クエリのトークナイズと記号の除去は組織者のノートブック
(experiment-transfer1-train.ipynb) と同じにする。重み付けは語の重みも飽和も持たず、
クエリの語ごとに 出現回数 ÷ 文書のトークン数 (Sudachi のトークン単位) を足す。
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
import pyterrier as pt
from sudachipy import dictionary, tokenizer

INDEX = "/work/transfer1/indexes/ntcir17-transfer/train"
TOPICS = Path("/work/transfer1/testcollections/ntcir/NTCIR-1/topics/topic0001-0083.utf8.jsonl")
RUN_FILE = Path("/work/tf-per-token/ntcir17-transfer/train/run.res.gz")
RESULTS_PER_TOPIC = 1000

# ノートブックの tokenize_topics と同じ記号。全角の記号はそのまま (組織者と同じ文字集合にする)
PUNCTUATION = re.compile(
    "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"  # noqa: RUF001
)


def tokenize(text, tokenizer_obj, mode):
    return " ".join(m.surface() for m in tokenizer_obj.tokenize(text, mode))


def read_topics():
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A
    rows = []
    with TOPICS.open(encoding="utf-8") as handle:
        for line in handle:
            topic = json.loads(line)
            query = PUNCTUATION.sub("", tokenize(topic["text"], tokenizer_obj, mode))
            rows.append((topic["query_id"], query))
    return pd.DataFrame(rows, columns=["qid", "query"])


def tf_per_token(key_frequency, posting, entry_stats, collection_stats):
    """クエリ側の出現回数 (key_frequency) は使わない。同じ語が繰り返されても 1 回と数える。"""
    return posting.getFrequency() / posting.getDocumentLength()


def main():
    pt.init(
        version=os.environ["TERRIER_VERSION"],
        helper_version=os.environ["TERRIER_HELPER_VERSION"],
    )
    index = pt.IndexFactory.of(INDEX)
    retrieve = pt.BatchRetrieve(index, wmodel=tf_per_token, num_results=RESULTS_PER_TOPIC)
    results = retrieve.transform(read_topics())
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    pt.io.write_results(results, str(RUN_FILE), format="trec", run_name="tf-per-token")
    print(f"run ファイル: {RUN_FILE} ({len(results)} 行, {results.qid.nunique()} トピック)")


if __name__ == "__main__":
    main()
