"""選び方から、記録用の表現を導けることを確かめる。"""

from __future__ import annotations

from hidden_subspace.experiment.describe import describe_dimensions
from hidden_subspace.selection import SelectAll, SelectRandom


def test_describes_a_selector_without_parameters() -> None:
    assert describe_dimensions(SelectAll()).as_record() == {"kind": "all"}


def test_describes_a_selector_with_parameters() -> None:
    choice = describe_dimensions(SelectRandom(count=100, seed=7))

    assert choice.as_record() == {"kind": "random", "count": 100, "seed": 7}
