"""隠れ状態から検索ベクトルを作る処理を確かめる。"""

from __future__ import annotations

import torch

from hidden_subspace.states import HiddenStates
from hidden_subspace.vectors import build_search_vectors


def states_of(values: torch.Tensor) -> HiddenStates:
    ids = tuple(f"d{i}" for i in range(values.shape[0]))
    return HiddenStates.of(values, item_ids=ids)


def test_concatenates_layers_in_order() -> None:
    """連結は層番号の昇順、その中で次元番号の昇順。

    内積はクエリと文書に同じ並べ替えを施しても変わらないため、この並びは検索の
    精度そのものには影響しない。固定するのは次の 2 つのため。

    - 保存した文書ベクトルと、後から作るクエリベクトルの並びが食い違わないこと。
      食い違っても例外は出ず、静かに性能だけが落ちる
    - ベクトル上の位置から (層, 次元) を辿れること

    値が単位長になるよう選び、正規化の影響を除いて並びだけを見る。
    """
    values = torch.tensor([[[0.6, 0.0], [0.0, 0.8]]])  # 連結すると長さ 1 になる

    built = build_search_vectors(states_of(values), normalize_layers=False)

    torch.testing.assert_close(built.values, torch.tensor([[0.6, 0.0, 0.0, 0.8]]))


def test_layer_normalization_equalizes_layer_contributions() -> None:
    """層ごとに揃えると各層の寄与が等しくなり、揃えないと大きい層が支配する。"""
    values = torch.zeros(1, 2, 2)
    values[0, 0] = torch.tensor([1.0, 0.0])  # ノルム 1
    values[0, 1] = torch.tensor([0.0, 100.0])  # ノルム 100

    without = build_search_vectors(states_of(values), normalize_layers=False)
    with_ = build_search_vectors(states_of(values), normalize_layers=True)

    assert without.values[0, :2].norm() < 0.02
    torch.testing.assert_close(with_.values[0, :2].norm(), with_.values[0, 2:].norm())


def test_layer_normalization_does_not_change_a_single_layer_result() -> None:
    """層が 1 つなら、層ごとに揃えても最終ベクトルは変わらない。"""
    values = torch.randn(5, 1, 4)

    without = build_search_vectors(states_of(values), normalize_layers=False)
    with_ = build_search_vectors(states_of(values), normalize_layers=True)

    torch.testing.assert_close(without.values, with_.values)


def test_reports_how_many_vectors_had_no_length() -> None:
    """静かに起きると性能が出ない原因に気づけないため、件数を残す。"""
    values = torch.zeros(3, 2, 3)
    values[2] = 1.0

    built = build_search_vectors(states_of(values), normalize_layers=True)

    assert built.zero_length_count == 2
    assert torch.isfinite(built.values).all()


def test_item_ids_are_carried_over() -> None:
    built = build_search_vectors(states_of(torch.randn(3, 2, 4)), normalize_layers=False)

    assert built.item_ids == ("d0", "d1", "d2")


def test_records_which_layers_and_dimensions_were_used() -> None:
    """どの構成で作られたベクトルかを、結果だけから辿れるようにする。"""
    states = states_of(torch.randn(3, 5, 8)).select_layers([1, 3]).select_dimensions([0, 2, 4])

    built = build_search_vectors(states, normalize_layers=False)

    assert built.layer_indices == (1, 3)
    assert built.dimension_indices == (0, 2, 4)
    assert built.values.shape == (3, 6)
