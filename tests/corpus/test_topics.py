"""トピックの読み込みを確かめる。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hidden_subspace.corpus.topics import read_topics

TOPIC = """<TOPIC q=0001>

<TITLE>
ロボット
</TITLE>

<DESCRIPTION>
自律移動ロボットについて
</DESCRIPTION>

<NARRATIVE>
自律移動ロボット自体の設計について書かれた文献。
</NARRATIVE>

</TOPIC>
"""


@pytest.fixture
def archive_of(
    write_archive: Callable[[Path, dict[str, str]], Path], tmp_path: Path
) -> Callable[[dict[str, str]], Path]:
    def build(members: dict[str, str]) -> Path:
        return write_archive(tmp_path / "TOPICS.TGZ", members)

    return build


def test_reads_the_identifier_title_and_description(
    archive_of: Callable[[dict[str, str]], Path],
) -> None:
    archive = archive_of({"topics/topic0001-0030": TOPIC})

    topics = read_topics(archive, ("topics/topic0001-0030",))

    assert len(topics) == 1
    assert topics[0].topic_id == "0001"
    assert topics[0].title == "ロボット"
    assert topics[0].description == "自律移動ロボットについて"


def test_reads_several_files_in_order(archive_of: Callable[[dict[str, str]], Path]) -> None:
    """トピックは 0001-0030 と 0031-0083 の 2 つのファイルに分かれている。"""
    second = TOPIC.replace("q=0001", "q=0031").replace("ロボット", "故障診断")
    archive = archive_of({"topics/topic0001-0030": TOPIC, "topics/topic0031-0083": second})

    topics = read_topics(archive, ("topics/topic0001-0030", "topics/topic0031-0083"))

    assert [t.topic_id for t in topics] == ["0001", "0031"]


def test_rejects_a_topic_without_a_title(archive_of: Callable[[dict[str, str]], Path]) -> None:
    body = TOPIC.replace("<TITLE>\nロボット\n</TITLE>\n", "")
    archive = archive_of({"topics/topic0001-0030": body})

    with pytest.raises(ValueError, match="TITLE"):
        read_topics(archive, ("topics/topic0001-0030",))
