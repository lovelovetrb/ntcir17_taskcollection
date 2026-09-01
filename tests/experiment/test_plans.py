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


def test_an_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="absent"):
        build_plans("absent", LAYERS, DIMENSIONS)


def test_names_are_listed_for_the_command_line() -> None:
    assert "layer-sweep" in plan_names()
