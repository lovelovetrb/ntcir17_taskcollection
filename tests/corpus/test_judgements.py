"""適合性判定の読み込みを確かめる。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hidden_subspace.corpus.judgements import read_judgements

MEMBER = "mlir/rel1_ntc1-j1_0001-0030"


@pytest.fixture
def archive_of(
    write_archive: Callable[[Path, dict[str, str]], Path], tmp_path: Path
) -> Callable[[str], Path]:
    def build(body: str) -> Path:
        return write_archive(tmp_path / "MLIR.TGZ", {MEMBER: body})

    return build


def test_reads_the_topic_document_and_grade(archive_of: Callable[[str], Path]) -> None:
    archive = archive_of("0001\tA\tgakkai-0000003395\t1\n")

    judgements = read_judgements(archive, (MEMBER,))

    assert len(judgements) == 1
    assert judgements[0].topic_id == "0001"
    assert judgements[0].document_id == "gakkai-0000003395"


def test_converts_grades_to_numbers() -> None:
    """A と B を適合、C を非適合として数える (ADR-0011)。"""
    from hidden_subspace.corpus.judgements import GRADE_VALUES

    assert GRADE_VALUES == {"A": 2, "B": 1, "C": 0}


def test_reads_grades_from_the_second_column(archive_of: Callable[[str], Path]) -> None:
    body = "0001\tA\tdoc1\t1\n0001\tB\tdoc2\t0\n0001\tC\tdoc3\t0\n"
    archive = archive_of(body)

    judgements = read_judgements(archive, (MEMBER,))

    assert [j.grade for j in judgements] == [2, 1, 0]


def test_accepts_lines_with_a_trailing_comment(archive_of: Callable[[str], Path]) -> None:
    """5 列目に日本語の判定者コメントが入る行があり、列数が 4/5 で揺れる。"""
    body = "0001\tA\tdoc1\t1\n0001\tB\tdoc2\t0\t複数データ制御についての言及無し\n"
    archive = archive_of(body)

    judgements = read_judgements(archive, (MEMBER,))

    assert [j.document_id for j in judgements] == ["doc1", "doc2"]


def test_rejects_an_unknown_grade(archive_of: Callable[[str], Path]) -> None:
    archive = archive_of("0001\tZ\tdoc1\t0\n")

    with pytest.raises(ValueError, match="Z"):
        read_judgements(archive, (MEMBER,))


def test_skips_blank_lines(archive_of: Callable[[str], Path]) -> None:
    archive = archive_of("0001\tA\tdoc1\t1\n\n")

    assert len(read_judgements(archive, (MEMBER,))) == 1
