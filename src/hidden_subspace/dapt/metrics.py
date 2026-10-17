"""MLM の評価指標: 損失、パープレキシティ、正解率、top-5 正解率 (ADR-0027)。

損失は `Trainer` が出す。パープレキシティは損失から求める。正解率と top-5 正解率は、
隠した位置 (`labels` が -100 でない位置) だけで数える。
"""

from __future__ import annotations

import math

import numpy as np
from torch import Tensor
from transformers import EvalPrediction, Trainer

TOP_K = 5
IGNORED_LABEL = -100


def keep_top_k(logits: Tensor, labels: Tensor) -> Tensor:
    """`preprocess_logits_for_metrics` に渡す。

    全語彙の出力を評価の間ためるとメモリが足りなくなるため、上位のトークン ID だけを残す。
    """
    return logits.topk(TOP_K, dim=-1).indices


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    top_k = np.asarray(prediction.predictions)
    labels = np.asarray(prediction.label_ids)
    masked = labels != IGNORED_LABEL
    candidates = top_k[masked]
    answers = labels[masked]
    return {
        "accuracy": float((candidates[:, 0] == answers).mean()),
        "top5_accuracy": float((candidates == answers[:, None]).any(axis=1).mean()),
    }


class MlmTrainer(Trainer):
    """評価の損失にパープレキシティを添えて記録する。

    `compute_metrics` は損失を受け取らず、`on_evaluate` のコールバックは記録の後に呼ばれる
    ため、記録の直前に足す。
    """

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for key, value in list(logs.items()):
            if key.startswith("eval_") and key.endswith("_loss"):
                logs[key.removesuffix("_loss") + "_perplexity"] = math.exp(value)
        super().log(logs, start_time)
