"""検証・テストデータの隠す位置を一度だけ決める (ADR-0027)。

学習中の評価のたびに隠す位置が変わると、ステップごとに別の問題を解かせることになり、
損失の推移を比べられない。学習と同じ collator を 1 回だけ通し、その結果を保存して使う。
"""

from __future__ import annotations

from collections.abc import Mapping

from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from hidden_subspace.dapt.splits import SEED

MLM_PROBABILITY = 0.15
MAX_LENGTH = 512


def fix_masks(texts: Mapping[str, str], tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """文書 ID ごとのテキストを、隠した `input_ids` と正解の `labels` にする。

    `labels` は隠した位置だけに元のトークンを持ち、それ以外は -100 になる。
    """
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm_probability=MLM_PROBABILITY, seed=SEED
    )
    encoded = tokenizer(list(texts.values()), truncation=True, max_length=MAX_LENGTH)
    rows: dict[str, list] = {"document_id": [], "input_ids": [], "attention_mask": [], "labels": []}
    for document_id, input_ids in zip(texts, encoded["input_ids"], strict=True):
        masked = collator([{"input_ids": input_ids}])
        rows["document_id"].append(document_id)
        rows["input_ids"].append(masked["input_ids"][0].tolist())
        rows["attention_mask"].append([1] * len(input_ids))
        rows["labels"].append(masked["labels"][0].tolist())
    return Dataset.from_dict(rows)
