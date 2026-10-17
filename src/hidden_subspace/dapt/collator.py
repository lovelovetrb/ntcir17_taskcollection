"""学習データは毎回隠し、検証・テストデータは保存した隠す位置のまま使う (ADR-0027)。

`Trainer` は学習用の collator を評価にも使う。MLM の collator をそのまま渡すと、隠す位置を
固定した検証・テストデータまで評価のたびに隠し直してしまう。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)

from hidden_subspace.dapt.masking import MLM_PROBABILITY


class MlmCollator:
    """`labels` を持つデータはパディングだけをし、持たないデータは隠す。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._mask = DataCollatorForLanguageModeling(tokenizer, mlm_probability=MLM_PROBABILITY)
        self._pad = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if "labels" in features[0]:
            return self._pad(list(features))
        return self._mask(list(features))
