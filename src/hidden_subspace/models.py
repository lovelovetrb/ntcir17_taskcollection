"""学習したモデルの置き場所と、それを指すモデルの識別子を決める。

学習したモデルは `<method>/<model_id>/<run_name>` という識別子で HF のモデルと同じように指す。
`<method>` は `dapt` か `dpr` で、置き場所は `cache/<method>/<model_id>/models/<run_name>/`
(ADR-0035)。識別子はキャッシュや記録の置き場所にもそのまま使う。
"""

from __future__ import annotations

from pathlib import Path

from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME

CACHE_ROOT = Path("cache")
METHODS = ("dapt", "dpr")
SAVED_FILES = (CONFIG_NAME, SAFE_WEIGHTS_NAME, TOKENIZER_CONFIG_FILE)
"""学習の終わりに保存されるファイル。ディレクトリは学習を始めた時点でできる。"""


def model_dir(method: str, model_id: str, run_name: str) -> Path:
    return CACHE_ROOT / method / model_id / "models" / run_name


def resolve_model(model_id: str) -> str:
    """`from_pretrained` に渡す値を返す。学習したモデルは置き場所に読み替える。"""
    method, _, rest = model_id.partition("/")
    if method not in METHODS:
        return model_id
    base_id, _, run_name = rest.rpartition("/")
    directory = model_dir(method, base_id, run_name)
    if not all((directory / name).is_file() for name in SAVED_FILES):
        raise FileNotFoundError(
            f"{directory} に学習したモデルがありません。"
            "識別子を確かめるか、学習を実行してください。"
        )
    return str(directory)
