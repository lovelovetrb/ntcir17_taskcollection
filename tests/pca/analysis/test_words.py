"""モデルに入る範囲で文章を打ち切り、語に分ける処理を確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import BertJapaneseTokenizer

from pca.analysis.common.words import new_tagger, segment, truncate

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB += ["ア", "##ル", "##ゴ", "##リ", "##ズ", "##ム", "を", "提", "##案", "する", "。"]
SENTENCE = "アルゴリズムを提案する。"
"""MeCab の語は アルゴリズム・を・提案・する・。 で、WordPiece は 6 + 1 + 2 + 1 + 1 = 11 個。"""


@pytest.fixture
def tokenizer(tmp_path: Path) -> BertJapaneseTokenizer:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return BertJapaneseTokenizer(
        str(vocab),
        word_tokenizer_type="mecab",
        subword_tokenizer_type="wordpiece",
        mecab_kwargs={"mecab_dic": "unidic_lite"},
    )


def test_truncated_text_fits_in_model_input(tokenizer: BertJapaneseTokenizer) -> None:
    text = SENTENCE * 100

    truncated = truncate(text, tokenizer, new_tagger())

    assert len(truncated) < len(text)
    assert len(tokenizer(truncated)["input_ids"]) <= 512


def test_truncation_does_not_cut_inside_word(tokenizer: BertJapaneseTokenizer) -> None:
    # 46 文で 506 個。47 文目の「アルゴリズム」(6 個) を足すと 512 個になり、
    # [CLS] と [SEP] を除いた 510 個に収まらない
    assert truncate(SENTENCE * 100, tokenizer, new_tagger()) == SENTENCE * 46


def test_counts_special_token_in_text_as_one(tokenizer: BertJapaneseTokenizer) -> None:
    # 通常のトークナイズでは [MASK] は 1 個で、本文は 506 + 3 + 1 = 510 個に収まる
    text = SENTENCE * 46 + "提案する[MASK]"

    assert truncate(text, tokenizer, new_tagger()) == text


def test_segment_excludes_symbols() -> None:
    # unidic-lite は「」と。を補助記号、Σ を記号にする
    words = segment("猫が「犬」を見た。Σ", new_tagger())

    assert [word.surface for word in words] == ["猫", "が", "犬", "を", "見", "た"]
