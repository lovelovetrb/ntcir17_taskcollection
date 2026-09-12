"""構成の一覧を作る処理を確かめる。"""

from __future__ import annotations

import pytest

from hidden_subspace.experiment.plans import build_plans, plan_names

LAYERS = (0, 1, 2)
DIMENSIONS = tuple(range(8))


def test_layer_sweep_uses_each_layer_alone() -> None:
    plans = build_plans("layer-sweep", LAYERS, DIMENSIONS)

    assert [p.layers for p in plans] == [(0,), (1,), (2,)]


def test_layer_sweep_keeps_every_dimension() -> None:
    plans = build_plans("layer-sweep", LAYERS, DIMENSIONS)

    assert all(p.dimensions(DIMENSIONS) == DIMENSIONS for p in plans)


def test_layer_sweep_does_not_normalize_layers() -> None:
    """層が 1 つのときは正規化の有無で結果が変わらない (ADR-0014)。"""
    plans = build_plans("layer-sweep", LAYERS, DIMENSIONS)

    assert all(p.normalize_layers is False for p in plans)


def test_layer_combinations_covers_every_subset_of_two_or_more() -> None:
    plans = build_plans("layer-combinations", LAYERS, DIMENSIONS)

    assert [p.layers for p in plans] == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]


def test_layer_combinations_leaves_out_single_layers() -> None:
    """層が 1 つの構成は層ごとの性能を測る実験が既に扱っている (ADR-0019)。"""
    plans = build_plans("layer-combinations", LAYERS, DIMENSIONS)

    assert all(len(p.layers) >= 2 for p in plans)


def test_layer_combinations_counts_8178_for_thirteen_layers() -> None:
    """13 層から 2 層以上を選ぶ組み合わせは 8,178 通り (ADR-0019)。"""
    plans = build_plans("layer-combinations", tuple(range(13)), DIMENSIONS)

    assert len(plans) == 8178
    assert len({p.layers for p in plans}) == 8178


def test_layer_combinations_keeps_every_dimension() -> None:
    plans = build_plans("layer-combinations", LAYERS, DIMENSIONS)

    assert all(p.dimensions(DIMENSIONS) == DIMENSIONS for p in plans)


def test_the_two_layer_combination_experiments_differ_only_in_normalization() -> None:
    """層ごとの正規化は実験として分ける (ADR-0019)。"""
    plain = build_plans("layer-combinations", LAYERS, DIMENSIONS)
    normalized = build_plans("layer-combinations-normalized", LAYERS, DIMENSIONS)

    assert [p.layers for p in plain] == [p.layers for p in normalized]
    assert all(p.normalize_layers is False for p in plain)
    assert all(p.normalize_layers is True for p in normalized)


def test_an_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent"):
        build_plans("absent", LAYERS, DIMENSIONS)


def test_names_are_listed_for_the_command_line() -> None:
    assert "layer-sweep" in plan_names()
