#!/usr/bin/env python3
"""トピックごとの適合性判定 (A/B/C) の件数を集計し、標準出力に表形式で表示する。

出力形式:
    TOPIC_NUM(TOPIC_TITLE) | Aの数 | Bの数 | Cの数

テストコレクションはリポジトリに含めない (利用許諾の対象)。配布された tarball を
展開せずに直接読む。既定の探索先は環境変数 NTCIR_DATA_DIR。

使い方:
    python sandbox/summarize_qrels_by_topic.py
    NTCIR_DATA_DIR=/data/ntcir python sandbox/summarize_qrels_by_topic.py
    python sandbox/summarize_qrels_by_topic.py <NTCIR-1 のディレクトリ>
"""

from __future__ import annotations

import os
import re
import sys
import tarfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

NTCIR_ENCODING = "euc_jp"

# C(非適合) も数えるため、2 値列ではなく grade 列を読む
GRADES: tuple[str, ...] = ("A", "B", "C")

# ADR-0009 / ADR-0011: 対象は mlir の ntc1-j1 に対する判定
QRELS_MEMBERS = (
    "mlir/rel1_ntc1-j1_0001-0030",
    "mlir/rel1_ntc1-j1_0031-0083",
)
TOPICS_MEMBERS = (
    "topics/topic0001-0030",
    "topics/topic0031-0083",
)
QRELS_ARCHIVE = "MLIR.TGZ"
TOPICS_ARCHIVE = "TOPICS.TGZ"

DEFAULT_DATA_DIR = Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17"))

_TOPIC_BLOCK = re.compile(r"<TOPIC\s+q=(\d+)>(.*?)</TOPIC>", re.DOTALL)
_TITLE_FIELD = re.compile(r"<TITLE>(.*?)</TITLE>", re.DOTALL)


@dataclass(frozen=True)
class TopicSummary:
    """1 トピック分の集計結果。"""

    topic_id: str
    title: str
    grade_counts: Counter[str]

    @property
    def judged_total(self) -> int:
        return sum(self.grade_counts.values())


def read_members(archive: Path, members: tuple[str, ...]) -> list[str]:
    """tarball 内の指定メンバーを展開せずに読み、EUC-JP としてデコードする。"""
    texts: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for name in members:
            handle = tar.extractfile(name)
            if handle is None:
                raise FileNotFoundError(f"{archive} にメンバー {name} がありません")
            texts.append(handle.read().decode(NTCIR_ENCODING))
    return texts


def load_topic_titles(topics_archive: Path) -> dict[str, str]:
    """トピックファイル群から トピックID -> TITLE の対応を作る。"""
    titles: dict[str, str] = {}
    for text in read_members(topics_archive, TOPICS_MEMBERS):
        for topic_id, block in _TOPIC_BLOCK.findall(text):
            match = _TITLE_FIELD.search(block)
            titles[topic_id] = " ".join(match.group(1).split()) if match else ""
    return titles


def count_grades_by_topic(qrels_archive: Path) -> dict[str, Counter[str]]:
    """qrels を読み、トピックごとにグレードの出現数を数える。

    1 行は TAB 区切りで (topic_id, grade, doc_id, binary_flag[, comment])。
    列数が 4/5 で揺れるため先頭 3 列だけを使う。
    """
    counts: dict[str, Counter[str]] = {}
    for text in read_members(qrels_archive, QRELS_MEMBERS):
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            topic_id, grade = fields[0], fields[1]
            counts.setdefault(topic_id, Counter())[grade] += 1
    return counts


def build_summaries(
    grade_counts: dict[str, Counter[str]], titles: dict[str, str]
) -> list[TopicSummary]:
    return [
        TopicSummary(topic_id, titles.get(topic_id, "(タイトル不明)"), counts)
        for topic_id, counts in sorted(grade_counts.items())
    ]


def display_width(text: str) -> int:
    """全角文字を 2 桁として数えた表示幅を返す。

    East Asian Width が Ambiguous の文字は多くの端末が半角で描画するため 1 桁とする。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "FW" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def render_table(summaries: list[TopicSummary]) -> str:
    labels = [f"{s.topic_id}({s.title})" for s in summaries]
    label_header = "TOPIC_NUM(TOPIC_TITLE)"
    label_width = max([display_width(label_header), *(display_width(x) for x in labels)])
    count_width = 8

    separator = "-+-".join(["-" * label_width, *("-" * count_width for _ in GRADES)])
    lines = [
        " | ".join([pad(label_header, label_width), *(g.rjust(count_width) for g in GRADES)]),
        separator,
    ]
    for label, summary in zip(labels, summaries, strict=True):
        cells = [str(summary.grade_counts.get(g, 0)).rjust(count_width) for g in GRADES]
        lines.append(" | ".join([pad(label, label_width), *cells]))

    totals: Counter[str] = Counter()
    for summary in summaries:
        totals.update(summary.grade_counts)
    lines.append(separator)
    lines.append(
        " | ".join(
            [
                pad(f"合計 ({len(summaries)} トピック)", label_width),
                *(str(totals.get(g, 0)).rjust(count_width) for g in GRADES),
            ]
        )
    )
    return "\n".join(lines)


def resolve_ntcir1_dir(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser()
    return (DEFAULT_DATA_DIR / "NTCIR-1").expanduser()


def main(argv: list[str]) -> int:
    ntcir1_dir = resolve_ntcir1_dir(argv)
    qrels_archive = ntcir1_dir / QRELS_ARCHIVE
    topics_archive = ntcir1_dir / TOPICS_ARCHIVE

    for archive in (qrels_archive, topics_archive):
        if not archive.is_file():
            print(f"見つかりません: {archive}", file=sys.stderr)
            print(
                "NTCIR_DATA_DIR を設定するか、NTCIR-1 のディレクトリを引数で渡してください。",
                file=sys.stderr,
            )
            return 1

    summaries = build_summaries(
        count_grades_by_topic(qrels_archive), load_topic_titles(topics_archive)
    )
    print(render_table(summaries))

    unexpected = {
        grade for summary in summaries for grade in summary.grade_counts if grade not in GRADES
    }
    if unexpected:
        print(f"\n警告: 未知のグレードを検出: {sorted(unexpected)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
