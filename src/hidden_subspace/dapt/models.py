"""学習したモデルの置き場所と、それを指すモデルの識別子を決める。

学習したモデルは `dapt/<model_id>/<run_name>` という識別子で HF のモデルと同じように指す。
識別子はキャッシュや記録の置き場所にもそのまま使う。
"""

from __future__ import annotations

from pathlib import Path

from hidden_subspace.dapt.data import CACHE_ROOT

PREFIX = "dapt/"


def model_dir(model_id: str, run_name: str) -> Path:
    return CACHE_ROOT / model_id / "models" / run_name


def resolve_model(model_id: str) -> str:
    """`from_pretrained` に渡す値を返す。学習したモデルは置き場所に読み替える。"""
    if not model_id.startswith(PREFIX):
        return model_id
    base_id, _, run_name = model_id.removeprefix(PREFIX).rpartition("/")
    directory = model_dir(base_id, run_name)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} に学習したモデルがありません。"
            "識別子を確かめるか、DAPT の学習を実行してください。"
        )
    return str(directory)
