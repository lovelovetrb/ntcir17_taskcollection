"""モデルを動かして、層ごとの隠れ状態を得る。

モデルは識別子で差し替える。層ごとの出力を得るために `output_hidden_states` を
必ず有効にする。入力は 512 トークンで打ち切る (ADR-0004)。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from hidden_subspace.encoding.pooling import mean_pool
from hidden_subspace.states import HiddenStates

MAX_TOKENS = 512
"""入力の打ち切り長 (ADR-0004)。対象モデルの位置埋め込みの上限でもある。"""


class Encoder:
    """事前学習済みモデルを保持し、テキストを層ごとの隠れ状態に変える。"""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        # from_pretrained の注釈は None を含むが、読み込みに失敗すれば例外が上がる
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"トークナイザを読み込めない: {model_id}")
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
        self.model.eval().to(device)

    @torch.no_grad()
    def encode(
        self, texts: Sequence[str], item_ids: Sequence[str], batch_size: int = 128
    ) -> HiddenStates:
        """テキストを層ごとの隠れ状態にする。"""
        if len(texts) != len(item_ids):
            raise ValueError(f"テキスト {len(texts)} 件と ID {len(item_ids)} 件が一致しない")
        pooled = torch.cat(list(self._encode_batches(texts, batch_size)))
        return HiddenStates.of(pooled, item_ids=item_ids)

    def _encode_batches(self, texts: Sequence[str], batch_size: int) -> Iterable[Tensor]:
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            hidden_states = torch.stack(self.model(**encoded).hidden_states)
            yield mean_pool(hidden_states, encoded["attention_mask"]).half().cpu()
