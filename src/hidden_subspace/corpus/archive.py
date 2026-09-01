"""配布された tarball から中身を読む。

テストコレクションは利用許諾の対象であり、読み取り専用でマウントされる。展開を
前提にせず、必要なメンバーを直接読む。
"""

from __future__ import annotations

import tarfile
from collections.abc import Sequence
from pathlib import Path

NTCIR_ENCODING = "euc_jp"
"""配布データの文字符号。"""

DECODE_ERRORS = "ignore"
"""復号できないバイトは捨てる。

配布データには元の入力や転記に由来する誤りが含まれており、`mlir/ntc1-j1` では
3.2 億バイト中 278 バイトが EUC-JP として解釈できない。NTCIR-17 Transfer-1 の
前処理も `iconv -c` で同じく捨てているため、それに合わせる (ADR-0011)。
"""


def read_members(archive: Path, members: Sequence[str]) -> list[str]:
    """tarball 内の指定したメンバーを、指定した順に読む。"""
    contents: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for name in members:
            handle = tar.extractfile(name)
            if handle is None:
                raise FileNotFoundError(f"{archive} にメンバー {name} がない")
            contents.append(handle.read().decode(NTCIR_ENCODING, DECODE_ERRORS))
    return contents
