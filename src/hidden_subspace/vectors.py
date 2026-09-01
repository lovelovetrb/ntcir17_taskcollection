"""隠れ状態から検索ベクトルを作る。

次元を選んだ後に層ごとに L2 正規化し、層を連結し、最後に検索ベクトルとして
L2 正規化する (ADR-0014, ADR-0017)。層ごとの正規化は実験軸であるため、呼び出し
ごとに明示する。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as functional
from torch import Tensor

from hidden_subspace.states import HiddenStates


@dataclass(frozen=True)
class SearchVectors:
    """検索に用いるベクトルと、それがどの構成から作られたか。"""

    values: Tensor
    item_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    dimension_indices: tuple[int, ...]
    zero_norm_count: int
    """すべての成分が 0 で、向きを持たないベクトルの件数。"""


def build_search_vectors(states: HiddenStates, *, normalize_layers: bool) -> SearchVectors:
    """隠れ状態を検索ベクトルにする。

    `normalize_layers` は層どうしのスケールを揃えるかどうか。残差接続により深い層
    ほどノルムが大きくなるため、揃えないと連結時に深い層が内積を支配する。

    すべての成分が 0 のベクトルは 0 のまま返る。正規化はノルムで割る操作であり、
    ノルムが 0 だと割れないため。実データでは滅多に起こらないが次元を極端に絞ると
    起こりうるので、1 件のために実験を止めず、件数だけを残す。
    """
    values = states.values.float()
    if normalize_layers:
        values = functional.normalize(values, dim=2)

    flattened = values.reshape(values.shape[0], -1)
    return SearchVectors(
        values=functional.normalize(flattened, dim=1),
        item_ids=states.item_ids,
        layer_indices=states.layer_indices,
        dimension_indices=states.dimension_indices,
        zero_norm_count=int((flattened.norm(dim=1) == 0).sum().item()),
    )
