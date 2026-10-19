"""追加事前学習を回す。

使う GPU は `CUDA_VISIBLE_DEVICES` で選ぶ。

使い方:
    CUDA_VISIBLE_DEVICES=0 python -m hidden_subspace.dapt
    CUDA_VISIBLE_DEVICES=0 python -m hidden_subspace.dapt --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hidden_subspace.dapt.config import DEFAULT_CONFIG_PATH, load_config
from hidden_subspace.dapt.train import train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_dir = train(load_config(arguments.config), arguments.data)
    print(f"モデルを保存した: {output_dir}")


if __name__ == "__main__":
    main()
