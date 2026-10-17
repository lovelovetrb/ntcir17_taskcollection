"""学習の設定の読み込みを確かめる。"""

from __future__ import annotations

from hidden_subspace.dapt.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads() -> None:
    load_config(DEFAULT_CONFIG_PATH)
