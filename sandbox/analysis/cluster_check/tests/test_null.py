"""比較に使う乱数データが、文書と同じ平均と共分散を持つことを確かめる。"""

from __future__ import annotations

import pytest
import torch

from analysis.cluster_check.null import gaussian_like


def test_has_the_same_mean_and_covariance_as_the_documents() -> None:
    generator = torch.Generator().manual_seed(0)
    mixing = torch.tensor([[2.0, 0.0, 0.0], [1.5, 0.5, 0.0], [0.0, -1.0, 0.3]])
    documents = torch.randn(5000, 3, generator=generator) @ mixing.T + torch.tensor(
        [1.0, -2.0, 0.5]
    )

    sampled = gaussian_like(documents, 200_000, generator=torch.Generator().manual_seed(1))

    assert sampled.shape == (200_000, 3)
    torch.testing.assert_close(sampled.mean(dim=0), documents.mean(dim=0), atol=0.02, rtol=0)
    torch.testing.assert_close(
        sampled.T.cov(correction=0),
        documents.T.double().cov(correction=0).float(),
        atol=0.03,
        rtol=0,
    )


def test_refuses_a_covariance_that_is_not_positive_definite() -> None:
    """値が一定の次元があると、共分散は正定値にならない。"""
    documents = torch.randn(100, 3, generator=torch.Generator().manual_seed(2))
    documents[:, 1] = 4.0

    with pytest.raises(ValueError, match="正定値"):
        gaussian_like(documents, 10, generator=torch.Generator().manual_seed(3))
