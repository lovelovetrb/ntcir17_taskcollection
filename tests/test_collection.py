"""コレクション一式の読み込みを確かめる。"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from hidden_subspace.collection import Ntcir1Paths, read_collection

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
            "mlir/rel1_ntc1-j1_0001-0030": QRELS,
            "mlir/rel1_ntc1-j1_0031-0083": "",
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
