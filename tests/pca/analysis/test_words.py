"""モデルに入る範囲で文章を打ち切り、語に分ける処理を確かめる。"""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import BertJapaneseTokenizer

from pca.analysis.common.words import load_tokenizer, new_tagger, segment, truncate

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


@pytest.mark.parametrize("special", ["[MASK]", "［ＭＡＳＫ］"])  # noqa: RUF001
def test_rejects_text_with_special_token(tokenizer: BertJapaneseTokenizer, special: str) -> None:
    # 全角は NFKC で [MASK] になる
    with pytest.raises(ValueError, match=r"\[MASK\]"):
        truncate(f"アルゴリズム{special}を提案する。", tokenizer, new_tagger())


def test_rejects_tokenizer_with_added_tokens(
    tokenizer: BertJapaneseTokenizer, tmp_path: Path
) -> None:
    tokenizer.add_tokens(["アルゴリズム"])
    tokenizer.save_pretrained(tmp_path / "added")

    with pytest.raises(ValueError, match="アルゴリズム"):
        load_tokenizer(str(tmp_path / "added"))


def test_segment_excludes_symbols() -> None:
    # unidic-lite は「」と。を補助記号、Σ を記号にする
    words = segment("猫が「犬」を見た。Σ", new_tagger())

    assert [word.surface for word in words] == ["猫", "が", "犬", "を", "見", "た"]
