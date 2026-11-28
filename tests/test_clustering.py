"""文書のベクトルを k-means で分ける部品を確かめる (ADR-0037)。

CI に GPU はないため、faiss の CPU 版で確かめる。GPU 版と CPU 版は同じ手順で、結果の差は
丸め誤差の範囲であることを PR #132 で確かめた。
"""

from __future__ import annotations

import torch

from hidden_subspace.clustering import fit_kmeans


def blobs(
    per_cluster: int, centers: torch.Tensor, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """互いに十分離れた中心の周りに、標準偏差 1 で点を置く。"""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(centers.shape[0]).repeat_interleave(per_cluster)
    noise = torch.randn(labels.shape[0], centers.shape[1], generator=generator)
    return centers[labels] + noise, labels


def same_partition(left: torch.Tensor, right: torch.Tensor) -> bool:
    """番号の付け方によらず、同じ分け方になっているか。"""
    pairs = {(int(a), int(b)) for a, b in zip(left, right, strict=True)}
    return len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})


def test_recovers_well_separated_clusters() -> None:
    centers = torch.eye(3, 8) * 50.0
    values, truth = blobs(200, centers)

    result = fit_kmeans(values, 3, seed=0, gpu=False)

    assert same_partition(result.labels, truth)


def test_centroids_are_the_means_of_all_assigned_points() -> None:
    """faiss は既定では 1 クラスタあたり 256 点までしか学習に使わない。全件を使っていれば、
    収束した中心は割り当てられた点の平均と一致する。"""
    centers = torch.eye(2, 4) * 50.0
    values, _ = blobs(1500, centers)

    result = fit_kmeans(values, 2, seed=0, gpu=False)

    for cluster in range(2):
        members = values[result.labels == cluster]
        torch.testing.assert_close(result.centroids[cluster], members.mean(dim=0))


def test_the_same_seed_gives_the_same_result() -> None:
    values = torch.randn(500, 6, generator=torch.Generator().manual_seed(1))

    first = fit_kmeans(values, 4, seed=7, gpu=False)
    second = fit_kmeans(values, 4, seed=7, gpu=False)

    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.centroids, second.centroids)


def test_one_cluster_is_the_mean_of_everything() -> None:
    values = torch.randn(100, 5, generator=torch.Generator().manual_seed(2))

    result = fit_kmeans(values, 1, seed=0, gpu=False)

    assert torch.equal(result.labels, torch.zeros(100, dtype=torch.long))
    torch.testing.assert_close(result.centroids[0], values.mean(dim=0))


def test_inertia_is_the_sum_of_squared_distances_to_the_assigned_centroids() -> None:
    values = torch.randn(300, 4, generator=torch.Generator().manual_seed(3))

    result = fit_kmeans(values, 3, seed=0, gpu=False)

    expected = ((values - result.centroids[result.labels]) ** 2).sum()
    assert abs(result.inertia - float(expected)) <= 1e-4 * float(expected)
