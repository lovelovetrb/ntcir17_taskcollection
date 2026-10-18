"""学習と評価でデータのまとめ方を切り替える collator を確かめる。"""

from __future__ import annotations

from typing import Any

import pytest
from transformers import BertTokenizer

from hidden_subspace.dapt.collator import MlmCollator
from hidden_subspace.dapt.masking import fix_masks


@pytest.fixture
def tokenizer() -> BertTokenizer:
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *(f"w{i}" for i in range(20))]
    return BertTokenizer(vocab={word: i for i, word in enumerate(words)})


TEXTS = {f"d{i}": " ".join(f"w{j % 20}" for j in range(i, i + 50 + i)) for i in range(4)}


def test_data_with_labels_is_not_masked_again(tokenizer: BertTokenizer) -> None:
    """検証・テストデータは、保存した隠す位置のまま使う。"""
    features: list[dict[str, Any]] = [
        {key: row[key] for key in ("input_ids", "attention_mask", "labels")}
        for row in fix_masks(TEXTS, tokenizer).to_list()
    ]

    batch = MlmCollator(tokenizer)(features)

    for row, feature in zip(batch["labels"].tolist(), features, strict=True):
        assert row[: len(feature["labels"])] == feature["labels"]


def test_data_without_labels_is_masked(tokenizer: BertTokenizer) -> None:
    features = [{"input_ids": ids} for ids in tokenizer(list(TEXTS.values()))["input_ids"]]

    batch = MlmCollator(tokenizer)(features)

    assert (batch["labels"] != -100).any()
