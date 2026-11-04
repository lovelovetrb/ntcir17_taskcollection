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


def test_scores_of_different_components_are_uncorrelated() -> None:
    """主成分を推定したのと同じ文書では、異なる主成分の得点どうしの相関が 0 になる。

    属性との r² を主成分をまたいで足すと、上位の主成分で回帰したときの決定係数になる。
    この読み方はこの性質に頼る。
    """
    generator = torch.Generator().manual_seed(0)
    mixing = torch.randn(8, 8, generator=generator)
    documents = torch.randn(200, 8, generator=generator) @ mixing

    components = estimate_common_components(documents, 3)

    correlations = torch.corrcoef(components.scores(documents).T)
    torch.testing.assert_close(correlations, torch.eye(3, dtype=correlations.dtype))
