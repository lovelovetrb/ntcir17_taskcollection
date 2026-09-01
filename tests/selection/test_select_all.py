"""すべて残す選び方。"""

from __future__ import annotations

from hidden_subspace.selection import SelectAll


def test_returns_the_given_numbers() -> None:
    assert SelectAll()((3, 7, 11)) == (3, 7, 11)
