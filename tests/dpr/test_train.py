"""学習の設定を読み、Tevatron に渡す引数を作る処理を確かめる。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from dpr.train import TrainingConfig, load_config, model_dir, tevatron_arguments

CONFIG_DIR = Path("src/dpr/config")


def config_of(pooling: str) -> TrainingConfig:
    return load_config(CONFIG_DIR / f"{pooling}.yaml")


def value_of(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_the_two_configs_differ_only_in_pooling() -> None:
    """CLS と mean を同じ条件で学習し、プーリングだけを比べる (ADR-0034)。"""
    cls, mean = config_of("cls"), config_of("mean")

    assert (cls.pooling, mean.pooling) == ("cls", "mean")
    assert dataclasses.replace(cls, pooling="mean") == mean


def test_the_configs_hold_the_values_decided_in_the_adrs() -> None:
    config = config_of("cls")

    assert config.model_id == "FacebookAI/xlm-roberta-large"
    assert (config.passage_max_len, config.query_max_len) == (512, 32)
    assert (config.train_group_size, config.batch_size, config.epochs) == (8, 8, 3)
    assert (config.learning_rate, config.warmup_ratio, config.seed) == (5e-5, 0.1, 42)
    assert (config.save_strategy, config.save_total_limit) == ("epoch", 1)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("model_id", "org/other-model"),
        ("pooling", "mean"),
        ("passage_max_len", 128),
        ("query_max_len", 64),
        ("train_group_size", 4),
        ("batch_size", 16),
        ("epochs", 1),
        ("learning_rate", 1e-5),
        ("warmup_ratio", 0.06),
        ("seed", 0),
    ],
)
def test_changing_a_training_value_changes_the_run_name(name: str, value: object) -> None:
    """同じ名前になると、別の設定で学習したモデルが同じ場所を指す。"""
    config = config_of("cls")

    assert dataclasses.replace(config, **{name: value}).run_name != config.run_name


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    """綴りを間違えた値が黙って既定値になるのを防ぐ。"""
    text = (CONFIG_DIR / "cls.yaml").read_text(encoding="utf-8") + "epoch: 1\n"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_the_model_is_saved_where_dpr_identifiers_point() -> None:
    """評価の側は `dpr/<model_id>/<run_name>` をこの場所に読み替える (ADR-0035)。"""
    assert model_dir("FacebookAI/xlm-roberta-large", "run") == Path(
        "cache/dpr/FacebookAI/xlm-roberta-large/models/run"
    )


def test_tevatron_receives_the_config_values(tmp_path: Path) -> None:
    config = config_of("mean")

    arguments = tevatron_arguments(config, tmp_path / "train.jsonl", tmp_path / "out")

    assert "--do_train" in arguments
    assert value_of(arguments, "--model_name_or_path") == "FacebookAI/xlm-roberta-large"
    assert value_of(arguments, "--pooling") == "mean"
    assert value_of(arguments, "--passage_max_len") == "512"
    assert value_of(arguments, "--per_device_train_batch_size") == "8"
    assert value_of(arguments, "--save_strategy") == "epoch"
    assert value_of(arguments, "--save_total_limit") == "1"
    assert value_of(arguments, "--dataset_path") == str(tmp_path / "train.jsonl")
    assert value_of(arguments, "--output_dir") == str(tmp_path / "out")
    assert value_of(arguments, "--run_name") == config.run_name
