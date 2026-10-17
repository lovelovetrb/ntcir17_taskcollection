"""学習の設定を yaml から読む。

yaml には変える頻度が高い値だけを置く。変えると検証・テストデータを作り直す必要がある値
(シード、検証の割合、隠す確率、系列長) はコードに固定する (ADR-0027)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WandbConfig:
    project: str
    entity: str | None = None
    """省略すると wandb にログインしているユーザーになる。"""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TrainingConfig:
    model_id: str
    epochs: Annotated[int, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    warmup_ratio: Annotated[float, Field(ge=0, lt=1)]
    """総ステップのうち warmup に充てる割合。"""
    wandb: WandbConfig

    @property
    def run_name(self) -> str:
        """wandb の run 名。モデルの保存先 `models/<run_name>/` の名前にもなる。

        学習の値を全部入れ、どれかを変えれば別の名前になるようにする。
        """
        model = self.model_id.rsplit("/", 1)[-1]
        return (
            f"{model}-ep{self.epochs}-bs{self.batch_size}"
            f"-lr{self.learning_rate:g}-wu{self.warmup_ratio:g}"
        )


def load_config(path: Path) -> TrainingConfig:
    return TypeAdapter(TrainingConfig).validate_python(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
