"""分析をまたいで使う道具の、グループ分けと相関を確かめる。"""

from __future__ import annotations

import pytest
import torch

from pca.analysis.common.correlation import correlations
from pca.analysis.common.groups import group_masks


def test_groups_documents_by_id_prefix() -> None:
    masks = group_masks(["gakkai-j-0000000001", "kaken-j-0986069700", "gakkai-j-0000000002"])

    assert masks["all"].tolist() == [True, True, True]
    assert masks["gakkai"].tolist() == [True, False, True]
    assert masks["kaken"].tolist() == [False, True, False]


def test_rejects_document_outside_groups() -> None:
    with pytest.raises(ValueError, match="other-0001"):
        group_masks(["gakkai-j-0000000001", "other-0001"])


def test_correlations_are_computed_per_component() -> None:
    values = torch.tensor([100.0, 200.0, 300.0, 400.0])
    scores = torch.stack([2 * values + 1, -0.5 * values], dim=1)

    torch.testing.assert_close(
        correlations(scores, values), torch.tensor([1.0, -1.0], dtype=torch.float64)
    )
