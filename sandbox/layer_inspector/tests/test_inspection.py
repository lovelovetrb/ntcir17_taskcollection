"""ビューアに表示する結果を、ファイルに保存して読み戻す処理を確かめる。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hidden_subspace.experiment.results import Configuration, DimensionChoice, TopicResult
from layer_inspector.documents import RankedDocument, TopicDocuments
from layer_inspector.inputs import ModelInputs
from layer_inspector.inspection import (
    Inspection,
    InspectionSettings,
    load_inspection,
    load_or_build_inspection,
    save_inspection,
)
from layer_inspector.selection import TopicChoice, TopicSelection

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


def result_of(topic_id: str, layer: int, score: float) -> TopicResult:
    configuration = Configuration(
        model_id=SETTINGS.model_id,
        layers=(layer,),
        dimensions=DimensionChoice(kind="all"),
        normalize_layers=False,
    )
    return TopicResult("layer-sweep", configuration, topic_id, {"nDCG@1000": score})


def inspection_of(settings: InspectionSettings = SETTINGS) -> Inspection:
    shallow = TopicChoice("0001", result_of("0001", 1, 0.5), result_of("0001", 12, 0.1), 0.4)
    deep = TopicChoice("0002", result_of("0002", 2, 0.1), result_of("0002", 11, 0.3), -0.2)
    documents = TopicDocuments(
        topic_id="0001",
        top=[RankedDocument("d1", shallow_rank=1, deep_rank=40, grade=None)],
        relevant_top=[RankedDocument("d2", shallow_rank=3, deep_rank=2, grade=2)],
        relevant_bottom=[],
    )
    return Inspection(
        settings=settings,
        selection=TopicSelection(shallow_favored=[shallow], deep_favored=[deep]),
        documents={"0001": documents},
        inputs=ModelInputs(
            queries={"0001": "[CLS] query [SEP]"},
            documents={"d1": "[CLS] title [UNK] [SEP]", "d2": "[CLS] 題名 抄録 [SEP]"},
        ),
    )


def test_a_saved_inspection_reads_back_unchanged(tmp_path: Path) -> None:
    """JSON では tuple が list に、判定のない grade が null になる。読み戻すと元の形に戻る。"""
    path = tmp_path / "inspection.json"

    save_inspection(inspection_of(), path)

    assert load_inspection(path) == inspection_of()


def test_builds_and_saves_when_there_is_no_file_and_reads_it_afterwards(tmp_path: Path) -> None:
    """結果を作るにはキャッシュの読み込みと GPU が要るため、2 度目以降は保存したものを読む。"""
    path = tmp_path / "inspection.json"
    calls: list[int] = []

    def build() -> Inspection:
        calls.append(1)
        return inspection_of()

    first = load_or_build_inspection(path, SETTINGS, build)
    second = load_or_build_inspection(path, SETTINGS, build)

    assert calls == [1]
    assert first == second == inspection_of()


def test_rebuilds_when_the_saved_file_was_made_with_other_settings(tmp_path: Path) -> None:
    """設定を変えたのに古いファイルを読むと、変える前の設定の結果を表示してしまう。"""
    path = tmp_path / "inspection.json"
    other = replace(SETTINGS, topic_count=3)
    save_inspection(inspection_of(other), path)

    loaded = load_or_build_inspection(path, SETTINGS, inspection_of)

    assert loaded.settings == SETTINGS
    assert load_inspection(path).settings == SETTINGS
