"""文書長を、512 トークンで打ち切った後のモデルのトークン数で数える (ADR-0030)。

エンコードと同じく、[CLS] と [SEP] を含めて `max_length=MAX_TOKENS` で打ち切る。
"""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Pool

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hidden_subspace.dapt.models import resolve_model
from hidden_subspace.encoding.encoder import MAX_TOKENS

_tokenizer: PreTrainedTokenizerBase | None = None


def _start_worker(model_id: str) -> None:
    global _tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolve_model(model_id))
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"{model_id} のトークナイザーを読み込めません。")
    _tokenizer = tokenizer


def _length_of(text: str) -> int:
    assert _tokenizer is not None
    return len(_tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"])


def document_lengths(texts: Sequence[str], model_id: str, processes: int) -> Tensor:
    with Pool(processes, initializer=_start_worker, initargs=(model_id,)) as pool:
        return torch.tensor(pool.map(_length_of, texts, chunksize=500))
