"""実験の構成と結果を表す。

記録は 1 件だけ取り出しても何の結果か分かる形にする。ディレクトリやファイル名を
情報の唯一の置き場所にすると、複数の実験を連結した時点で区別がつかなくなる。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DimensionChoice:
    """次元の選び方。選び方ごとに持つ引数が異なる。

    選ばれた番号そのものは持たない。`kind` と引数から再現できるため。
    """

    kind: str
    parameters: Mapping[str, int] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {"kind": self.kind, **dict(self.parameters)}


@dataclass(frozen=True)
class Configuration:
    """1 つの検索ベクトルを決める条件。

    `layers` は元のモデルにおける層番号。組み合わせは 2^13 通りありうるが、
    1 つの構成が持つ番号は最大 13 個に収まるため、選び方ではなく結果そのものを
    持つ。次元は 1 つの構成が 768 個持ちうるため、選び方だけを持つ。
    """

    model_id: str
    layers: tuple[int, ...]
    dimensions: DimensionChoice
    normalize_layers: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "layers": list(self.layers),
            "dimensions": self.dimensions.as_record(),
            "normalize_layers": self.normalize_layers,
        }


@dataclass(frozen=True)
class ConfigurationResult:
    """構成 1 つに対する、全トピックを平均した結果。"""

    experiment: str
    configuration: Configuration
    metrics: Mapping[str, float]

    def as_record(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            **self.configuration.as_record(),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class TopicResult:
    """構成 1 つに対する、トピック 1 件の結果。"""

    experiment: str
    configuration: Configuration
    topic_id: str
    metrics: Mapping[str, float]

    def as_record(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            **self.configuration.as_record(),
            "topic_id": self.topic_id,
            "metrics": dict(self.metrics),
        }
