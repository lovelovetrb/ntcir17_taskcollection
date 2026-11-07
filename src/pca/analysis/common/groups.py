"""NTCIR-2 の文書を、文書 ID の先頭で学会発表と科研費報告書に分ける (ADR-0030、ADR-0032)。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

PREFIXES = {"gakkai": "gakkai-", "kaken": "kaken-"}
"""グループ名と、そのグループの文書 ID の先頭。NTCIR-1 由来の文書も `gakkai-` で始まる。"""
GROUPS = ("all", *PREFIXES)


def group_masks(document_ids: Sequence[str]) -> dict[str, Tensor]:
    """グループごとに、そのグループに入る文書を True にした `[文書数]` を返す。"""
    masks = {
        group: torch.tensor([document_id.startswith(prefix) for document_id in document_ids])
        for group, prefix in PREFIXES.items()
    }
    covered = torch.stack(list(masks.values())).any(dim=0)
    if not covered.all():
        outside = document_ids[int((~covered).nonzero()[0])]
        raise ValueError(
            f"文書 {outside} はどのグループにも入りません。NTCIR-2 の文書 ID か確認してください。"
        )
    return {"all": covered, **masks}
