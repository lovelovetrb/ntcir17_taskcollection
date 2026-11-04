"""文書集合の平均と主成分の推定を確かめる。"""

from __future__ import annotations

import torch

from pca.core.components import estimate_common_components


def test_directions_are_orthonormal() -> None:
    """主成分の向きは長さ 1 で、互いに直交する。

    得点 (x - x̄) · u_k を主成分の向きへの射影として読むことと、除く処理
    centered - centered @ U @ U.T が射影の除去になることは、どちらもこの性質に頼る。
    """
    documents = torch.randn(200, 8, generator=torch.Generator().manual_seed(0))

    components = estimate_common_components(documents, 3)

    gram = components.directions.T @ components.directions
    torch.testing.assert_close(gram, torch.eye(3, dtype=gram.dtype))
