"""選んだ点から全件までの距離で求めるシルエット係数を、scikit-learn と照らし合わせる。"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import silhouette_samples

from analysis.cluster_check.silhouette import sampled_silhouette


def points_and_labels(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(60, 5, generator=generator)
    values[:20] += 3.0
    labels = torch.arange(3).repeat_interleave(20)
    return values, labels


def test_matches_scikit_learn_when_every_point_is_chosen() -> None:
    values, labels = points_and_labels(0)

    ours = sampled_silhouette(values, labels, torch.arange(60), k=3, batch=7)

    expected = silhouette_samples(values.double().numpy(), labels.numpy())
    np.testing.assert_allclose(ours.numpy(), expected, rtol=1e-9, atol=1e-12)


def test_chosen_points_get_the_same_value_as_with_every_point_chosen() -> None:
    """距離は全件までで測るため、選んだ点の値は全件を選んだときの値と同じになる。"""
    values, labels = points_and_labels(1)
    chosen = torch.tensor([3, 17, 25, 59])

    ours = sampled_silhouette(values, labels, chosen, k=3, batch=2)

    expected = silhouette_samples(values.double().numpy(), labels.numpy())[chosen.numpy()]
    np.testing.assert_allclose(ours.numpy(), expected, rtol=1e-9, atol=1e-12)


def test_a_point_alone_in_its_cluster_gets_zero() -> None:
    """scikit-learn と同じく、1 点だけのクラスタの点は 0 にする。"""
    values = torch.tensor([[0.0], [0.1], [0.2], [10.0]])
    labels = torch.tensor([0, 0, 0, 1])

    ours = sampled_silhouette(values, labels, torch.arange(4), k=2, batch=4)

    assert ours[3].item() == 0.0
    expected = silhouette_samples(values.double().numpy(), labels.numpy())
    np.testing.assert_allclose(ours.numpy(), expected, rtol=1e-9, atol=1e-12)
