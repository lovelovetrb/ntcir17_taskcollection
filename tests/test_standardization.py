"""隠れ状態を層・次元ごとに z スコアに直す処理を確かめる (ADR-0037)。"""

from __future__ import annotations

import pytest
import torch

from hidden_subspace import standardization
from hidden_subspace.standardization import fit_standardization
from hidden_subspace.states import HiddenStates


def states_of(values: torch.Tensor) -> HiddenStates:
    ids = tuple(f"d{i}" for i in range(values.shape[0]))
    return HiddenStates.of(values, item_ids=ids)


def test_fit_returns_mean_and_population_std_per_layer_and_dimension() -> None:
    """標準偏差は n で割る。文書は数十万件あり n - 1 との差は無視できるが、値を固定しておく。"""
    values = torch.tensor(
        [
            [[1.0, 10.0], [0.0, -2.0]],
            [[3.0, 10.0], [4.0, 2.0]],
            [[5.0, 16.0], [8.0, 6.0]],
        ]
    )

    fitted = fit_standardization(states_of(values))

    torch.testing.assert_close(
        fitted.mean, torch.tensor([[3.0, 12.0], [4.0, 2.0]], dtype=torch.float64)
    )
    expected_std = torch.tensor(
        [[(8 / 3) ** 0.5, 8**0.5], [(32 / 3) ** 0.5, (32 / 3) ** 0.5]], dtype=torch.float64
    )
    torch.testing.assert_close(fitted.std, expected_std)


def test_applying_to_fitted_documents_gives_zero_mean_and_unit_std() -> None:
    values = torch.randn(200, 3, 4) * torch.tensor([1.0, 5.0, 0.1, 30.0]) + 7.0
    states = states_of(values)

    standardized = fit_standardization(states).apply(states).values.double()

    torch.testing.assert_close(standardized.mean(dim=0), torch.zeros(3, 4, dtype=torch.float64))
    torch.testing.assert_close(
        standardized.std(dim=0, unbiased=False), torch.ones(3, 4, dtype=torch.float64)
    )


def test_applying_to_narrowed_states_matches_narrowing_after_applying() -> None:
    """検索では層と次元を絞ってから直す。絞る前に直したときと同じ値になる必要がある。"""
    states = states_of(torch.randn(20, 4, 5))
    standardization = fit_standardization(states)

    narrowed_first = standardization.apply(states.select_layers([3, 1]).select_dimensions([4, 0]))
    applied_first = standardization.apply(states).select_layers([3, 1]).select_dimensions([4, 0])

    torch.testing.assert_close(narrowed_first.values, applied_first.values)
    assert narrowed_first.layer_indices == (3, 1)
    assert narrowed_first.dimension_indices == (4, 0)


def test_fit_refuses_a_dimension_with_zero_std() -> None:
    values = torch.randn(10, 2, 3)
    values[:, 1, 2] = 4.0

    with pytest.raises(ValueError, match="標準偏差が 0"):
        fit_standardization(states_of(values))


def test_chunked_fit_and_apply_match_the_unchunked_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """区切って計算しても、区切らないときと同じ値になる。"""
    states = states_of(torch.randn(7, 3, 4) * 5.0 + 2.0)
    whole = fit_standardization(states)
    expected = whole.apply(states).values

    monkeypatch.setattr(standardization, "CHUNK_ELEMENTS", 2 * 3 * 4)
    chunked = fit_standardization(states)

    torch.testing.assert_close(chunked.mean, whole.mean)
    torch.testing.assert_close(chunked.std, whole.std)
    torch.testing.assert_close(chunked.apply(states).values, expected)


def test_fit_refuses_an_empty_document_set() -> None:
    with pytest.raises(ValueError, match="文書が 1 件もありません"):
        fit_standardization(states_of(torch.zeros(0, 2, 3)))
