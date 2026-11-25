"""組織者の DPR_X_TT を学び直す (ADR-0036)。

学習データ (`cache/dpr/data/train.jsonl`) がなければ、取った mMARCO のファイルから作る。
使う GPU は `CUDA_VISIBLE_DEVICES` で 1 枚だけ選ぶ。

使い方:
    CUDA_VISIBLE_DEVICES=0 uv run python -m dpr --config src/dpr/config/cls.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dpr.data import TRAINING_FILE, prepare_training_data
from dpr.download import DOWNLOAD_DIR
from dpr.train import load_config, train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    data_path = prepare_training_data(DOWNLOAD_DIR, TRAINING_FILE)
    output_dir = train(config, data_path)
    print(f"モデルを保存した: {output_dir}")


if __name__ == "__main__":
    main()
