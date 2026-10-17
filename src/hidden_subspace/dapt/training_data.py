"""学習データを作る (ADR-0027)。

NTCIR-1 の文書から、検証に回した文書を除いた残りをトークン化する。隠すのは学習中に
collator が毎回行うため、ここでは隠さない。キャッシュには置かず、学習のたびに作る。
"""

from __future__ import annotations

from collections.abc import Set
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from hidden_subspace.collection import collection_paths, read_collection
from hidden_subspace.corpus.text import document_text
from hidden_subspace.dapt.masking import MAX_LENGTH
from hidden_subspace.dapt.splits import exclude


def build_training_data(
    data_root: Path, validation_ids: Set[str], tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    collection = read_collection(collection_paths("ntcir1", data_root))
    documents = exclude(collection.documents, validation_ids)
    encoded = tokenizer(
        [document_text(d) for d in documents], truncation=True, max_length=MAX_LENGTH
    )
    return Dataset.from_dict({"input_ids": encoded["input_ids"]})
