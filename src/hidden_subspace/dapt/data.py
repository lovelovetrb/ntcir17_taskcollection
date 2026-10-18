"""学習・検証・テストデータを作り、`cache/dapt/<model_id>/data/` に置く (ADR-0027、ADR-0028)。

学習データは分割・トークン化した状態で置き、隠すのは学習中に行う。検証・テストデータは隠した
状態で置く。あれば読み、なければ作る。`splits.py`・`tokens.py`・`masking.py` の固定値を
変えたときは `data/` を消してから作り直す。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from hidden_subspace.collection import Ntcir1Paths, Ntcir2Paths, collection_paths
from hidden_subspace.corpus.documents import Document, read_documents
from hidden_subspace.corpus.text import document_text
from hidden_subspace.dapt.masking import fix_masks
from hidden_subspace.dapt.splits import sample_test, split_validation
from hidden_subspace.dapt.tokens import tokenize

CACHE_ROOT = Path("cache/dapt")


@dataclass(frozen=True)
class DaptData:
    training: Dataset
    validation: Dataset
    test: Dataset


def data_dir(model_id: str) -> Path:
    return CACHE_ROOT / model_id / "data"


def load_or_build(path: Path, build: Callable[[], Dataset]) -> Dataset:
    if path.exists():
        dataset = load_from_disk(str(path))
        assert isinstance(dataset, Dataset)
        return dataset
    dataset = build()
    dataset.save_to_disk(str(path))
    return dataset


def prepare_data(data_root: Path, model_id: str, tokenizer: PreTrainedTokenizerBase) -> DaptData:
    directory = data_dir(model_id)

    @cache
    def split() -> tuple[list[Document], list[Document]]:
        return split_validation(_ntcir1_documents(data_root))

    training = load_or_build(directory / "train", lambda: tokenize(_texts(split()[0]), tokenizer))
    validation = load_or_build(directory / "val", lambda: fix_masks(_texts(split()[1]), tokenizer))
    test = load_or_build(
        directory / "test",
        lambda: fix_masks(_texts(_test_documents(data_root, len(validation))), tokenizer),
    )
    return DaptData(training=training, validation=validation, test=test)


def _ntcir1_documents(data_root: Path) -> list[Document]:
    paths = collection_paths("ntcir1", data_root)
    assert isinstance(paths, Ntcir1Paths)
    return read_documents(paths.root / paths.documents_archive, paths.documents_member)


def _test_documents(data_root: Path, count: int) -> list[Document]:
    """`ntc2-j1g` と `ntc2-j1k` はどちらも NTCIR-1 と重ならない (`ntcir2-unique`)。"""
    paths = collection_paths("ntcir2", data_root)
    assert isinstance(paths, Ntcir2Paths)
    archive = paths.ntcir2_root / paths.documents_archive
    gakkai = read_documents(archive, paths.gakkai_member)
    kaken = read_documents(archive, paths.kaken_member, title_tag="PJNM")
    return sample_test(gakkai, kaken, count)


def _texts(documents: list[Document]) -> dict[str, str]:
    return {d.document_id: document_text(d) for d in documents}
