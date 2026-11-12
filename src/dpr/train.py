"""設定ファイルを読み、Tevatron の学習を回す (ADR-0036)。

Tevatron の学習の入口 (`tevatron.retriever.driver.train`) を別のプロセスで動かし、設定ファイルの値を
引数で渡す。設定ファイルにない値は Tevatron の既定値のままになる。学習したモデルとトークナイザは
`cache/dpr/<model_id>/models/<run_name>/` に置く。wandb のファイルも同じ場所に置く。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

CACHE_ROOT = Path("cache/dpr")
DATASET_CACHE_DIR = CACHE_ROOT / "data" / "datasets"
"""Tevatron が jsonl を読むときに作る `datasets` のキャッシュ。"""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WandbConfig:
    project: str
    entity: str | None = None
    """省略すると wandb にログインしているユーザーになる。"""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TrainingConfig:
    model_id: str
    pooling: Literal["cls", "mean"]
    passage_max_len: Annotated[int, Field(gt=0)]
    query_max_len: Annotated[int, Field(gt=0)]
    train_group_size: Annotated[int, Field(gt=0)]
    """1 クエリあたりの文書の数。適合文書 1 件と負例 `train_group_size` - 1 件。"""
    batch_size: Annotated[int, Field(gt=0)]
    """GPU 1 枚あたりのクエリの数。"""
    epochs: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    warmup_ratio: Annotated[float, Field(ge=0, lt=1)]
    seed: int
    attn_implementation: str
    save_strategy: Literal["no", "steps", "epoch"]
    save_total_limit: Annotated[int, Field(gt=0)] | None
    wandb: WandbConfig

    @property
    def run_name(self) -> str:
        """wandb の run 名。モデルの保存先 `models/<run_name>/` の名前にもなる。

        学習の結果を変える値を全部入れ、どれかを変えれば別の名前になるようにする。
        Attention の実装、チェックポイントの残し方、wandb の送り先は、学習したモデルを変えない
        ので入れない。
        """
        model = self.model_id.rsplit("/", 1)[-1]
        return (
            f"{model}-{self.pooling}-p{self.passage_max_len}-q{self.query_max_len}"
            f"-g{self.train_group_size}-ep{self.epochs}-bs{self.batch_size}"
            f"-lr{self.learning_rate:g}-wu{self.warmup_ratio:g}-s{self.seed}"
        )


def load_config(path: Path) -> TrainingConfig:
    return TypeAdapter(TrainingConfig).validate_python(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def model_dir(model_id: str, run_name: str) -> Path:
    return CACHE_ROOT / model_id / "models" / run_name


def tevatron_arguments(config: TrainingConfig, data_path: Path, output_dir: Path) -> list[str]:
    """Tevatron の学習の入口に渡す引数。"""
    arguments = {
        "--output_dir": str(output_dir),
        "--model_name_or_path": config.model_id,
        "--pooling": config.pooling,
        "--attn_implementation": config.attn_implementation,
        "--dataset_name": "json",
        "--dataset_path": str(data_path),
        "--dataset_cache_dir": str(DATASET_CACHE_DIR),
        "--passage_max_len": str(config.passage_max_len),
        "--query_max_len": str(config.query_max_len),
        "--train_group_size": str(config.train_group_size),
        "--per_device_train_batch_size": str(config.batch_size),
        "--num_train_epochs": str(config.epochs),
        "--learning_rate": str(config.learning_rate),
        "--warmup_ratio": str(config.warmup_ratio),
        "--seed": str(config.seed),
        "--save_strategy": config.save_strategy,
        "--report_to": "wandb",
        "--run_name": config.run_name,
    }
    if config.save_total_limit is not None:
        arguments["--save_total_limit"] = str(config.save_total_limit)
    return ["--do_train", *(item for pair in arguments.items() for item in pair)]


def train(config: TrainingConfig, data_path: Path) -> Path:
    """学習したモデルの置き場所を返す。

    同じ run 名のモデルがすでにあれば、上書きせずに止める。
    """
    output_dir = model_dir(config.model_id, config.run_name)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} がすでにあります。"
            "設定の値を変えるか、このディレクトリを消してから実行してください。"
        )
    output_dir.mkdir(parents=True)
    environment = {
        **os.environ,
        "WANDB_PROJECT": config.wandb.project,
        "WANDB_DIR": str(output_dir),
    }
    if config.wandb.entity is not None:
        environment["WANDB_ENTITY"] = config.wandb.entity
    command = [
        sys.executable,
        "-m",
        "tevatron.retriever.driver.train",
        *tevatron_arguments(config, data_path, output_dir),
    ]
    subprocess.run(command, env=environment, check=True)
    return output_dir
