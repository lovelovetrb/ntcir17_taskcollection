"""モデルの識別子を読み込み先に読み替える処理を確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hidden_subspace.dapt.models import resolve_model

RUN_NAME = "bert-base-japanese-v3-ep3-bs128-lr5e-05-wu0.06"
DIRECTORY = Path("cache/dapt/cl-tohoku/bert-base-japanese-v3/models") / RUN_NAME
SAVED_FILES = ("config.json", "model.safetensors", "tokenizer_config.json")


def test_resolve_model_keeps_a_huggingface_id() -> None:
    assert resolve_model("cl-tohoku/bert-base-japanese-v3") == "cl-tohoku/bert-base-japanese-v3"


def test_resolve_model_points_a_dapt_id_to_the_trained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    DIRECTORY.mkdir(parents=True)
    for name in SAVED_FILES:
        (DIRECTORY / name).touch()

    resolved = resolve_model(f"dapt/cl-tohoku/bert-base-japanese-v3/{RUN_NAME}")

    assert Path(resolved) == DIRECTORY


def test_resolve_model_rejects_a_dapt_id_without_a_trained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """読み替えずに渡すと HF Hub へ取りに行き、原因の分かりにくいエラーになる。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        resolve_model(f"dapt/cl-tohoku/bert-base-japanese-v3/{RUN_NAME}")


def test_resolve_model_rejects_a_run_stopped_before_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """学習を始めるとディレクトリはできるが、モデルは学習が終わってから保存される。"""
    monkeypatch.chdir(tmp_path)
    (DIRECTORY / "wandb").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        resolve_model(f"dapt/cl-tohoku/bert-base-japanese-v3/{RUN_NAME}")
