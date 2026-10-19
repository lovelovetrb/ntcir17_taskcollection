"""モデルの識別子を読み込み先に読み替える処理を確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hidden_subspace.dapt.models import resolve_model

RUN_NAME = "bert-base-japanese-v3-ep3-bs128-lr5e-05-wu0.06"


def test_resolve_model_keeps_a_huggingface_id() -> None:
    assert resolve_model("cl-tohoku/bert-base-japanese-v3") == "cl-tohoku/bert-base-japanese-v3"


def test_resolve_model_points_a_dapt_id_to_the_trained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = Path("cache/dapt/cl-tohoku/bert-base-japanese-v3/models") / RUN_NAME
    directory.mkdir(parents=True)

    resolved = resolve_model(f"dapt/cl-tohoku/bert-base-japanese-v3/{RUN_NAME}")

    assert Path(resolved) == directory


def test_resolve_model_rejects_a_dapt_id_without_a_trained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """読み替えずに渡すと HF Hub へ取りに行き、原因の分かりにくいエラーになる。"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        resolve_model(f"dapt/cl-tohoku/bert-base-japanese-v3/{RUN_NAME}")
