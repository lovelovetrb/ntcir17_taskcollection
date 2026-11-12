"""文書長を、512 トークンで打ち切った後のモデルのトークン数で数える (ADR-0030)。

エンコードと同じく、[CLS] と [SEP] を含めて `max_length=MAX_TOKENS` で打ち切る。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hidden_subspace.encoding.encoder import MAX_TOKENS
from hidden_subspace.models import resolve_model

_tokenizer: PreTrainedTokenizerBase | None = None


def _length_of(model_id: str, text: str) -> int:
    # Pool の initializer で読み込むと、失敗したときに Pool がワーカーを作り直し続けて
    # 終わらない。最初のタスクで読み込み、失敗を親プロセスに返す
    global _tokenizer
    if _tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(resolve_model(model_id))
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"{model_id} のトークナイザーを読み込めません。")
        _tokenizer = tokenizer
    return len(_tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"])


def document_lengths(texts: Sequence[str], model_id: str, processes: int) -> Tensor:
    with Pool(processes) as pool:
        return torch.tensor(pool.map(partial(_length_of, model_id), texts, chunksize=500))
