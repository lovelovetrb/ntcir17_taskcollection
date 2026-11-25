"""シルエット係数の記録を読み書きする。

層ごとに 1 ファイルにする。層を分けて複数の GPU で回したときに、書き込みがぶつからない。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Source = Literal["documents", "gaussian"]
DOCUMENTS: Source = "documents"
GAUSSIAN: Source = "gaussian"


@dataclass(frozen=True)
class SilhouetteRecord:
    """1 つのデータ・層・k についての結果。

    `repeat` は乱数データを作り直した回の番号で、文書では None。
    """

    collection: str
    model_id: str
    pooling: str
    layer: int
    k: int
    source: Source
    repeat: int | None
    silhouette: float
    inertia: float
    sample_size: int
    seed: int
    iterations: int
    restarts: int


def layer_file(directory: Path, layer: int) -> Path:
    return directory / f"layer_{layer:02d}.jsonl"


def write_layer(directory: Path, layer: int, records: list[SilhouetteRecord]) -> Path:
    """その層の記録を置き換える。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = layer_file(directory, layer)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return path


def read_records(directory: Path) -> list[SilhouetteRecord]:
    """層の番号順に、すべての層の記録を読む。"""
    paths = sorted(directory.glob("layer_*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"シルエット係数の記録がありません: {directory}\n"
            "先に python -m analysis.cluster_check.run を実行してください。"
        )
    return [
        SilhouetteRecord(**json.loads(line))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
