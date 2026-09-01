"""文書の読み込みを確かめる。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hidden_subspace.corpus.documents import read_documents

RECORD = """<REC>
<ACCN>gakkai-0000000001</ACCN>
<TITL TYPE="kanji">電気回路演習用CAIとその改良</TITL>
<AUPK TYPE="kanji">小野 敏夫</AUPK>
<ABST TYPE="kanji"><ABST.P>大学等での基礎的な演習を支援する。</ABST.P>\
<ABST.P>改良を行った。</ABST.P></ABST>
<SOCN TYPE="kanji">電気学会</SOCN>
</REC>
"""


@pytest.fixture
def archive_of(
    write_archive: Callable[[Path, dict[str, str]], Path], tmp_path: Path
) -> Callable[[str], Path]:
    def build(body: str) -> Path:
        return write_archive(tmp_path / "MLIR.TGZ", {"mlir/ntc1-j1": body})

    return build


def test_reads_the_identifier_title_and_abstract(archive_of: Callable[[str], Path]) -> None:
    documents = read_documents(archive_of(RECORD), "mlir/ntc1-j1")

    assert len(documents) == 1
    assert documents[0].document_id == "gakkai-0000000001"
    assert documents[0].title == "電気回路演習用CAIとその改良"


def test_joins_abstract_paragraphs(archive_of: Callable[[str], Path]) -> None:
    """抄録は段落に分かれている。タグを外して 1 続きにする。"""
    documents = read_documents(archive_of(RECORD), "mlir/ntc1-j1")

    assert documents[0].abstract == "大学等での基礎的な演習を支援する。 改良を行った。"


def test_keeps_the_order_of_the_source(archive_of: Callable[[str], Path]) -> None:
    body = RECORD + RECORD.replace("0000000001", "0000000002")

    documents = read_documents(archive_of(body), "mlir/ntc1-j1")

    assert [d.document_id for d in documents] == [
        "gakkai-0000000001",
        "gakkai-0000000002",
    ]


def test_rejects_a_record_without_a_title(archive_of: Callable[[str], Path]) -> None:
    """対象の文書集合は全件が題名と抄録を持つ。欠けていれば別の集合を読んでいる。"""
    body = RECORD.replace('<TITL TYPE="kanji">電気回路演習用CAIとその改良</TITL>\n', "")

    with pytest.raises(ValueError, match="TITL"):
        read_documents(archive_of(body), "mlir/ntc1-j1")


def test_rejects_a_record_without_an_abstract(archive_of: Callable[[str], Path]) -> None:
    body = RECORD.replace(
        '<ABST TYPE="kanji"><ABST.P>大学等での基礎的な演習を支援する。</ABST.P>'
        "<ABST.P>改良を行った。</ABST.P></ABST>\n",
        "",
    )

    with pytest.raises(ValueError, match="ABST"):
        read_documents(archive_of(body), "mlir/ntc1-j1")
