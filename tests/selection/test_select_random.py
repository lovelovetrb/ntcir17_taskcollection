"""無作為に選ぶ選び方が満たすべき性質を確かめる。"""

from __future__ import annotations

import pytest

from hidden_subspace.selection import SelectRandom

AVAILABLE = tuple(range(20))


def test_the_same_seed_gives_the_same_result() -> None:
    """同じ種で同じ結果が出なければ、種を変えて分散を測る実験が成立しない。"""
    first = SelectRandom(count=5, seed=42)(AVAILABLE)
    second = SelectRandom(count=5, seed=42)(AVAILABLE)

    assert first == second


def test_returns_the_requested_count() -> None:
    assert len(SelectRandom(count=5, seed=0)(AVAILABLE)) == 5


def test_selects_a_subset_without_duplicates() -> None:
    selected = SelectRandom(count=5, seed=0)(AVAILABLE)

    assert len(set(selected)) == len(selected)
    assert set(selected) <= set(AVAILABLE)


def test_result_is_sorted() -> None:
    """内積は次元の順序に依存しないため検索結果は変わらないが、ベクトル上の位置から
    (層, 次元) を辿る際に順序が予測できる。"""
    selected = SelectRandom(count=5, seed=0)(AVAILABLE)

    assert list(selected) == sorted(selected)


def test_selects_from_the_given_numbers_not_from_zero() -> None:
    """絞り込み済みの番号列を渡された場合、その中から選ぶ。"""
    available = (3, 7, 11, 13)
    selected = SelectRandom(count=2, seed=0)(available)

    assert set(selected) <= set(available)


def test_requesting_more_than_available_is_rejected() -> None:
    with pytest.raises(ValueError):
        SelectRandom(count=21, seed=0)(AVAILABLE)
