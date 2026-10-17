"""隠す位置を固定した検証・テストデータを作り、`cache/dapt/<model_id>/data/` に置く (ADR-0027)。

あれば読み、なければ作る。`splits.py` と `masking.py` の固定値を変えたときは `data/` を
消してから作り直す。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from hidden_subspace.collection import Ntcir2Paths, collection_paths, read_collection
from hidden_subspace.corpus.documents import Document, read_documents
from hidden_subspace.corpus.text import document_text
from hidden_subspace.dapt.masking import fix_masks
from hidden_subspace.dapt.splits import sample_test, split_validation

CACHE_ROOT = Path("cache/dapt")


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


def prepare_evaluation_data(
    data_root: Path, model_id: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[Dataset, Dataset]:
    """検証データとテストデータを返す。"""
    directory = data_dir(model_id)
    validation = load_or_build(
        directory / "val", lambda: _masked(_validation_documents(data_root), tokenizer)
    )
    test = load_or_build(
        directory / "test",
        lambda: _masked(_test_documents(data_root, len(validation)), tokenizer),
    )
    return validation, test


def _validation_documents(data_root: Path) -> list[Document]:
    collection = read_collection(collection_paths("ntcir1", data_root))
    _, validation = split_validation(collection.documents)
    return validation


def _test_documents(data_root: Path, count: int) -> list[Document]:
    """`ntc2-j1g` と `ntc2-j1k` はどちらも NTCIR-1 と重ならない (`ntcir2-unique`)。"""
    paths = collection_paths("ntcir2", data_root)
    assert isinstance(paths, Ntcir2Paths)
    archive = paths.ntcir2_root / paths.documents_archive
    gakkai = read_documents(archive, paths.gakkai_member)
    kaken = read_documents(archive, paths.kaken_member, title_tag="PJNM")
    return sample_test(gakkai, kaken, count)


def _masked(documents: list[Document], tokenizer: PreTrainedTokenizerBase) -> Dataset:
    return fix_masks({d.document_id: document_text(d) for d in documents}, tokenizer)
