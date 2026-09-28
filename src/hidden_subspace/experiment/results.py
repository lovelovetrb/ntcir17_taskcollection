"""実験の構成と結果を表す。

記録は 1 件だけ取り出しても何の結果か分かる形にする。ディレクトリやファイル名を
情報の唯一の置き場所にすると、複数の実験を連結した時点で区別がつかなくなる。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hidden_subspace.evaluation import Evaluation


@dataclass(frozen=True)
class DimensionChoice:
    """次元の選び方。選び方ごとに持つ引数が異なる。

    選ばれた番号そのものは持たない。`kind` と引数から再現できるため。
    """

    kind: str
    parameters: Mapping[str, int] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {"kind": self.kind, **dict(self.parameters)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DimensionChoice:
        """`kind` の横に並べた引数を `parameters` に戻す。"""
        parameters = {name: value for name, value in record.items() if name != "kind"}
        return cls(kind=record["kind"], parameters=parameters)


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

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Configuration:
        """記録では list になっている層番号を tuple に戻す。"""
        return cls(
            model_id=record["model_id"],
            layers=tuple(record["layers"]),
            dimensions=DimensionChoice.from_record(record["dimensions"]),
            normalize_layers=record["normalize_layers"],
        )


@dataclass(frozen=True)
class ConfigurationResult:
    """構成 1 つに対する、全トピックを平均した結果。

    向きを持たないベクトルの件数を併せて持つ。0 ベクトルどうしはスコアが等しく
    並び順で上位が決まるため、指標だけでは結果を読めない。次元を絞るほど増える。
    """

    experiment: str
    configuration: Configuration
    metrics: Mapping[str, float]
    zero_norm_queries: int
    zero_norm_documents: int

    def as_record(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            **self.configuration.as_record(),
            "metrics": dict(self.metrics),
            "zero_norm": {
                "queries": self.zero_norm_queries,
                "documents": self.zero_norm_documents,
            },
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

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TopicResult:
        """構成の項目は同じ階層に並べて書くため、同じ記録から読む。"""
        return cls(
            experiment=record["experiment"],
            configuration=Configuration.from_record(record),
            topic_id=record["topic_id"],
            metrics=dict(record["metrics"]),
        )


def results_of_evaluation(
    experiment: str,
    configuration: Configuration,
    evaluation: Evaluation,
    *,
    zero_norm_queries: int = 0,
    zero_norm_documents: int = 0,
) -> tuple[ConfigurationResult, list[TopicResult]]:
    """指標の計算結果を、構成 1 件の記録とトピックごとの記録にする。

    `zero_norm_queries` と `zero_norm_documents` は向きを持たないベクトルの件数で、
    ベクトルで検索しない場合は 0 のままでよい。
    """
    summary = ConfigurationResult(
        experiment=experiment,
        configuration=configuration,
        metrics=evaluation.macro_average(),
        zero_norm_queries=zero_norm_queries,
        zero_norm_documents=zero_norm_documents,
    )
    per_topic = [
        TopicResult(
            experiment=experiment,
            configuration=configuration,
            topic_id=topic_id,
            metrics={name: float(values[row].item()) for name, values in evaluation.scores.items()},
        )
        for row, topic_id in enumerate(evaluation.topic_ids)
    ]
    return summary, per_topic
