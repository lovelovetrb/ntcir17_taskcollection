"""NTCIR-1 の文書を学習と検証に分け、NTCIR-2 にだけある文書からテストを抜き出す (ADR-0027)。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch.utils.data import Dataset, random_split

from hidden_subspace.corpus.documents import Document

SEED = 42
VALIDATION_RATIO = 0.05


def split_validation(documents: Sequence[Document]) -> tuple[list[Document], list[Document]]:
    """学習と検証に分ける。

    件数は `random_split` に任せる。各側の件数を切り捨て、余りを学習側から順に足すため、
    割り切れないときは学習側が切り上がる。
    """
    training, validation = random_split(
        cast(Dataset[int], range(len(documents))),
        [1 - VALIDATION_RATIO, VALIDATION_RATIO],
        generator=torch.Generator().manual_seed(SEED),
    )
    return [documents[i] for i in training.indices], [documents[i] for i in validation.indices]


def sample_test(
    gakkai: Sequence[Document], kaken: Sequence[Document], count: int
) -> list[Document]:
    """学会発表と科研費報告書から `count` の半分ずつ抜き出す。

    半分に満たない側は全件を使う。
    """
    generator = torch.Generator().manual_seed(SEED)
    sampled: list[Document] = []
    for documents in (gakkai, kaken):
        order = torch.randperm(len(documents), generator=generator)[: count // 2]
        sampled.extend(documents[i] for i in order.tolist())
    return sampled
