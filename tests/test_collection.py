"""コレクション一式の読み込みを確かめる。"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from hidden_subspace.collection import Ntcir1Paths, Ntcir2Paths, read_collection

DOCUMENT = """<REC>
<ACCN>gakkai-0000000001</ACCN>
<TITL TYPE="kanji">題名</TITL>
<ABST TYPE="kanji"><ABST.P>抄録</ABST.P></ABST>
</REC>
"""
TOPIC = """<TOPIC q=0001>
<TITLE>
ロボット
</TITLE>
<DESCRIPTION>
自律移動ロボットについて
</DESCRIPTION>
</TOPIC>
"""
QRELS = "0001\tA\tgakkai-0000000001\t1\n"


def build(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in members.items():
            data = text.encode("euc_jp")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_reads_documents_topics_and_qrels_together(tmp_path: Path) -> None:
    build(
        tmp_path / "MLIR.TGZ",
        {
            "mlir/ntc1-j1": DOCUMENT,
            "mlir/rel2_ntc1-j1_0001-0030": QRELS,
            "mlir/rel2_ntc1-j1_0031-0083": "",
        },
    )
    build(
        tmp_path / "TOPICS.TGZ",
        {"topics/topic0001-0030": TOPIC, "topics/topic0031-0083": ""},
    )

    collection = read_collection(Ntcir1Paths(tmp_path))

    assert [d.document_id for d in collection.documents] == ["gakkai-0000000001"]
    assert [t.topic_id for t in collection.topics] == ["0001"]
    assert collection.qrels == {"0001": {"gakkai-0000000001": 2}}


GAKKAI_DOCUMENT = """<REC>
<ACCN>gakkai-j-0000339502</ACCN>
<TITL TYPE="kanji">学会の題名</TITL>
<ABST TYPE="kanji"><ABST.P>学会の抄録</ABST.P></ABST>
</REC>
"""
KAKEN_DOCUMENT = """<REC>
<ACCN>kaken-j-0851100100</ACCN>
<PJNM TYPE="kanji">研究課題名</PJNM>
<ABST TYPE="kanji"><ABST.P>科研の抄録</ABST.P></ABST>
</REC>
"""
NTCIR2_TOPIC = TOPIC.replace("q=0001", "q=0101")
NTCIR2_QRELS = "0101\tS\tgakkai-j-0000000001\t1\t\n0101\tB\tkaken-j-0851100100\t1\t\n"


def test_reads_ntcir2_with_the_ntcir1_documents(tmp_path: Path) -> None:
    """NTCIR-2 の文書集合は NTCIR-1 の文書を ID を書き換えて含む (ADR-0026)。"""
    ntcir1_root = tmp_path / "NTCIR-1"
    ntcir2_root = tmp_path / "NTCIR-2"
    ntcir1_root.mkdir()
    ntcir2_root.mkdir()
    build(ntcir1_root / "MLIR.TGZ", {"mlir/ntc1-j1": DOCUMENT})
    build(
        ntcir2_root / "j-docs.tgz",
        {"j-docs/ntc2-j1g": GAKKAI_DOCUMENT, "j-docs/ntc2-j1k": KAKEN_DOCUMENT},
    )
    build(ntcir2_root / "topics.tgz", {"topics/topic-j0101-0149": NTCIR2_TOPIC})
    build(ntcir2_root / "rels.tgz", {"rels/rel2_ntc2-j2_0101-0149": NTCIR2_QRELS})

    collection = read_collection(Ntcir2Paths(ntcir1_root, ntcir2_root))

    assert [d.document_id for d in collection.documents] == [
        "gakkai-j-0000000001",
        "gakkai-j-0000339502",
        "kaken-j-0851100100",
    ]
    assert [d.title for d in collection.documents] == ["題名", "学会の題名", "研究課題名"]
    assert [t.topic_id for t in collection.topics] == ["0101"]
    assert collection.qrels == {"0101": {"gakkai-j-0000000001": 3, "kaken-j-0851100100": 1}}
