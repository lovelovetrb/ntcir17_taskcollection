"""MLM の評価指標の計算を確かめる。"""

from __future__ import annotations

import numpy as np
import torch
from transformers import EvalPrediction

from hidden_subspace.dapt.metrics import TOP_K, compute_metrics, keep_top_k


def test_accuracy_counts_only_masked_positions() -> None:
    """隠していない位置 (-100) は、予測が外れていても数えない。"""
    predictions = np.array([[[7, 1, 2, 3, 4], [9, 7, 2, 3, 4], [0, 0, 0, 0, 0]]])
    labels = np.array([[7, 7, -100]])

    metrics = compute_metrics(EvalPrediction(predictions=predictions, label_ids=labels))

    assert metrics == {"accuracy": 0.5, "top5_accuracy": 1.0}


def test_keep_top_k_returns_token_ids_instead_of_every_logit() -> None:
    logits = torch.randn(2, 3, 100)

    kept = keep_top_k(logits, torch.zeros(2, 3, dtype=torch.long))

    assert kept.shape == (2, 3, TOP_K)
    assert torch.equal(kept[..., 0], logits.argmax(dim=-1))
