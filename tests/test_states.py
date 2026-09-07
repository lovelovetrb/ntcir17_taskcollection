"""隠れ状態の domain model が満たすべき性質を確かめる。"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace.states import HiddenStates


@pytest.fixture
def states() -> HiddenStates:
    """3 文書 x 5 層 x 4 次元。値は (文書, 層, 次元) が判別できるように作る。"""
    values = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4)
    return HiddenStates.of(values, item_ids=("a", "b", "c"))


def test_of_assigns_original_indices(states: HiddenStates) -> None:
    assert states.layer_indices == (0, 1, 2, 3, 4)
    assert states.dimension_indices == (0, 1, 2, 3)
    assert states.item_ids == ("a", "b", "c")


def test_selecting_layers_keeps_dimensions(states: HiddenStates) -> None:
    selected = states.select_layers([1, 3])

    assert selected.values.shape == (3, 2, 4)
    assert selected.layer_indices == (1, 3)
    assert selected.dimension_indices == states.dimension_indices


def test_selecting_dimensions_keeps_layers(states: HiddenStates) -> None:
    selected = states.select_dimensions([0, 2])

    assert selected.values.shape == (3, 5, 2)
    assert selected.dimension_indices == (0, 2)
    assert selected.layer_indices == states.layer_indices


def test_selections_commute(states: HiddenStates) -> None:
    """層と次元は独立した軸なので、絞る順序によらず同じ結果になる。"""
    layers_first = states.select_layers([1, 3]).select_dimensions([0, 2])
    dimensions_first = states.select_dimensions([0, 2]).select_layers([1, 3])

    assert torch.equal(layers_first.values, dimensions_first.values)
    assert layers_first.layer_indices == dimensions_first.layer_indices
    assert layers_first.dimension_indices == dimensions_first.dimension_indices


def test_selection_carries_original_indices(states: HiddenStates) -> None:
    """絞り込みを重ねても、元のモデルにおける番号が保たれる。"""
    selected = states.select_layers([1, 2, 3]).select_layers([2, 3]).select_layers([3])

    assert selected.layer_indices == (3,)


def test_selection_picks_the_right_values(states: HiddenStates) -> None:
    selected = states.select_layers([1, 3]).select_dimensions([0, 2])

    expected = states.values[:, [1, 3], :][:, :, [0, 2]]
    assert torch.equal(selected.values, expected)


def test_selection_does_not_mutate_the_source(states: HiddenStates) -> None:
    """PyTorch のテンソルは可変なので、絞り込みが元を壊さないことを保証する。"""
    before = states.values.clone()

    selected = states.select_layers([1, 3])
    selected.values[:] = -1.0

    assert torch.equal(states.values, before)


def test_selecting_an_unknown_layer_is_rejected(states: HiddenStates) -> None:
    with pytest.raises(ValueError):
        states.select_layers([99])


def test_selecting_an_unknown_dimension_is_rejected(states: HiddenStates) -> None:
    with pytest.raises(ValueError):
        states.select_dimensions([99])


def test_item_ids_must_match_the_number_of_rows() -> None:
    values = torch.zeros(3, 5, 4)
    with pytest.raises(ValueError):
        HiddenStates.of(values, item_ids=("a", "b"))


def test_moving_to_a_device_keeps_the_identifiers(states: HiddenStates) -> None:
    """置き場所を変えても、何であるかは変わらない。"""
    moved = states.to("cpu")

    assert moved.item_ids == states.item_ids
    assert moved.layer_indices == states.layer_indices
    assert moved.dimension_indices == states.dimension_indices
    assert torch.equal(moved.values, states.values)
