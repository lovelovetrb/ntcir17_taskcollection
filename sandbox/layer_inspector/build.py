"""ビューアが表示する結果ファイルを作る。

    make layer-inspector-build

記録・キャッシュ・GPU・トークナイザを使う。表示側はここで書き出したファイルを読むだけで
動く (ADR-0021)。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from hidden_subspace.collection import collection_paths
from layer_inspector.inspection import (
    InspectionSettings,
    build_inspection,
    load_or_build_inspection,
)

SETTINGS = InspectionSettings(
    model_id="cl-tohoku/bert-base-japanese-v3",
    experiment="layer-sweep",
    metric="nDCG@1000",
    shallow_layers=(1, 2),
    deep_layers=(11, 12),
    topic_count=5,
    top_count=10,
    relevant_count=5,
)
COLLECTION = collection_paths(
    "ntcir1", Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser()
)
CACHE_ROOT = Path("cache/ntcir1")
RESULTS_ROOT = Path("results/ntcir1")
RESULT_FILE = RESULTS_ROOT / SETTINGS.model_id / "layer-inspector" / "inspection.json"


def main() -> None:
    """結果ファイルがなければ作り、条件が同じものがあればそのまま使う。"""
    # 記録と同じ順位にするには GPU で検索し直す。CPU では記録との照合で止まる。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_or_build_inspection(
        RESULT_FILE,
        SETTINGS,
        lambda: build_inspection(SETTINGS, COLLECTION, CACHE_ROOT, RESULTS_ROOT, device),
    )
    print(f"結果ファイル: {RESULT_FILE}")


if __name__ == "__main__":
    main()
