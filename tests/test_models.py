"""モデルの識別子を読み込み先に読み替える処理を確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hidden_subspace.models import resolve_model

BASE_ID = "cl-tohoku/bert-base-japanese-v3"
RUN_NAME = "bert-base-japanese-v3-ep3-bs128-lr5e-05-wu0.06"
SAVED_FILES = ("config.json", "model.safetensors", "tokenizer_config.json")
METHODS = pytest.mark.parametrize("method", ["dapt", "dpr"])


def directory_of(method: str) -> Path:
    return Path("cache") / method / BASE_ID / "models" / RUN_NAME


def test_resolve_model_keeps_a_huggingface_id() -> None:
    assert resolve_model(BASE_ID) == BASE_ID


@METHODS
def test_resolve_model_points_a_trained_id_to_the_trained_model(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = directory_of(method)
    directory.mkdir(parents=True)
    for name in SAVED_FILES:
        (directory / name).touch()

    resolved = resolve_model(f"{method}/{BASE_ID}/{RUN_NAME}")

    assert Path(resolved) == directory


@METHODS
def test_resolve_model_rejects_a_trained_id_without_a_trained_model(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """読み替えずに渡すと HF Hub へ取りに行き、原因の分かりにくいエラーになる。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        resolve_model(f"{method}/{BASE_ID}/{RUN_NAME}")


@METHODS
def test_resolve_model_rejects_a_run_stopped_before_saving(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """学習を始めるとディレクトリはできるが、モデルは学習が終わってから保存される。"""
    monkeypatch.chdir(tmp_path)
    (directory_of(method) / "wandb").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        resolve_model(f"{method}/{BASE_ID}/{RUN_NAME}")
