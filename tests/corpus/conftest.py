"""読み込みのテストで使う、小さな tarball を組み立てる補助。"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

WriteArchive = Callable[[Path, dict[str, str]], Path]


@pytest.fixture
def write_archive() -> WriteArchive:
    """EUC-JP で符号化した中身を持つ tar.gz を作る。"""

    def write(path: Path, members: dict[str, str]) -> Path:
        with tarfile.open(path, "w:gz") as tar:
            for name, text in members.items():
                data = text.encode("euc_jp")
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return path

    return write
