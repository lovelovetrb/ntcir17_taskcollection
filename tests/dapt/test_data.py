"""検証・テストデータの作り方を確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset
from transformers import BertTokenizer

from hidden_subspace.corpus.documents import Document
from hidden_subspace.dapt.evaluation_data import load_or_build
from hidden_subspace.dapt.masking import fix_masks
from hidden_subspace.dapt.splits import exclude, sample_test


def documents(prefix: str, count: int) -> list[Document]:
    return [Document(f"{prefix}-{i}", "題名", "抄録") for i in range(count)]


def genres(sampled: list[Document]) -> list[str]:
    return [d.document_id.split("-")[0] for d in sampled]


def test_sample_test_takes_half_of_the_count_from_each_genre() -> None:
    sampled = sample_test(documents("gakkai", 10), documents("kaken", 10), count=6)

    assert genres(sampled).count("gakkai") == 3
    assert genres(sampled).count("kaken") == 3


def test_sample_test_uses_every_document_of_a_genre_smaller_than_its_share() -> None:
    sampled = sample_test(documents("gakkai", 2), documents("kaken", 10), count=6)

    assert genres(sampled).count("gakkai") == 2
    assert genres(sampled).count("kaken") == 3


def test_exclude_leaves_no_validation_document_in_training() -> None:
    all_documents = documents("gakkai", 10)
    validation_ids = {"gakkai-1", "gakkai-4"}

    training = exclude(all_documents, validation_ids)

    assert len(training) == 8
    assert not {d.document_id for d in training} & validation_ids


@pytest.fixture
def tokenizer() -> BertTokenizer:
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *(f"w{i}" for i in range(20))]
    return BertTokenizer(vocab={word: i for i, word in enumerate(words)})


def test_fix_masks_gives_the_same_masks_every_time(tokenizer: BertTokenizer) -> None:
    texts = {f"d{i}": " ".join(f"w{j % 20}" for j in range(i, i + 100)) for i in range(5)}

    first = fix_masks(texts, tokenizer)
    second = fix_masks(texts, tokenizer)

    assert first["input_ids"] == second["input_ids"]
    assert first["labels"] == second["labels"]
    assert any(label != -100 for labels in first["labels"] for label in labels)


def test_load_or_build_reads_the_saved_data_without_building(tmp_path: Path) -> None:
    path = tmp_path / "val"
    saved = Dataset.from_dict({"document_id": ["a", "b"]})
    load_or_build(path, lambda: saved)

    def fail() -> Dataset:
        raise AssertionError("作り直した")

    assert load_or_build(path, fail)["document_id"] == ["a", "b"]
