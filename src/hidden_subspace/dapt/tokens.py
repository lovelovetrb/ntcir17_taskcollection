"""文書テキストをトークン化する (ADR-0027)。"""

from __future__ import annotations

from collections.abc import Mapping

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

MAX_LENGTH = 512


def tokenize(texts: Mapping[str, str], tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """文書 ID ごとのテキストを `input_ids` にする。512 トークンで切る (ADR-0004)。"""
    encoded = tokenizer(list(texts.values()), truncation=True, max_length=MAX_LENGTH)
    return Dataset.from_dict({"document_id": list(texts), "input_ids": encoded["input_ids"]})
