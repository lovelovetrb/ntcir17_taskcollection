"""mMARCO の日本語版の学習に使う 3 つのファイルを取り、`cache/dpr/data/mmarco/` に置く (ADR-0036)。

取ったファイルは手を加えずに置く。置き場所の下は mMARCO のリポジトリと同じ並びになる。

    uv run python -m dpr.download
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "unicamp-dl/mmarco"
DOWNLOAD_DIR = Path("cache/dpr/data/mmarco")
TRIPLES = "data/triples.train.ids.small.tsv"
QUERIES = "data/google/queries/train/japanese_queries.train.tsv"
COLLECTION = "data/google/collections/japanese_collection.tsv"
FILES = (TRIPLES, QUERIES, COLLECTION)


def download(directory: Path = DOWNLOAD_DIR) -> list[Path]:
    """取ったファイルの置き場所を返す。すでにあるファイルは取り直さない。"""
    return [
        Path(hf_hub_download(REPOSITORY, name, repo_type="dataset", local_dir=directory))
        for name in FILES
    ]


def main() -> None:
    for path in download():
        print(path)


if __name__ == "__main__":
    main()
