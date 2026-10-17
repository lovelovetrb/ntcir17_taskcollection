"""学習の設定の読み込みを確かめる。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hidden_subspace.dapt.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads() -> None:
    load_config(DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "cl-nagoya/unsup-simcse-ja-base"},
        {"epochs": 10},
        {"batch_size": 64},
        {"learning_rate": 1.0e-4},
        {"warmup_ratio": 0.1},
    ],
)
def test_run_name_changes_with_each_training_value(change: dict[str, object]) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    assert replace(config, **change).run_name != config.run_name
