"""学習データは毎回隠し、検証・テストデータは保存した隠す位置のまま使う (ADR-0027、ADR-0028)。

`Trainer` は学習用の collator を評価にも使う。MLM の collator をそのまま渡すと、隠す位置を
固定した検証・テストデータまで評価のたびに隠し直してしまう。

学習データを隠す collator には専用の乱数を持たせる。全体の乱数を使うと、ほかの処理が乱数を
1 回でも多く引いたときに、以降の隠し方がすべてずれる。
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
from hidden_subspace.dapt.splits import SEED


class MlmCollator:
    """`labels` を持つデータはパディングだけをし、持たないデータは隠す。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._mask = DataCollatorForLanguageModeling(
            tokenizer, mlm_probability=MLM_PROBABILITY, seed=SEED
        )
        self._pad = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if "labels" in features[0]:
            return self._pad(list(features))
        return self._mask(list(features))
