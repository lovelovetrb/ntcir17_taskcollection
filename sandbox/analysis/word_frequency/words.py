"""モデルに入る範囲の文章を、MeCab (unidic-lite) で語に分ける (ADR-0029)。"""

from __future__ import annotations

import importlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import unidic_lite
from transformers import BertJapaneseTokenizer

MAX_TOKENS = 512
BUDGET = MAX_TOKENS - 2
"""[CLS] と [SEP] を除いて、文章のトークンに使える数。"""

EXCLUDED_POS = ("記号", "補助記号")
NO_SUBCLASS = "*"
"""中分類・小分類がないときに MeCab が返す値。"""


@dataclass(frozen=True)
class Word:
    """語を表層形・品詞の大分類・中分類の組で区別する。中分類がない品詞では `pos2` が None。"""

    surface: str
    pos: str
    """品詞の大分類 (名詞、助詞など)。"""
    pos2: str | None


@dataclass(frozen=True)
class TaggedWord:
    """語と、MeCab が付けた品詞の小分類。小分類では語を区別しない。"""

    word: Word
    pos3: str


def truncate(text: str, tokenizer: BertJapaneseTokenizer) -> str:
    """モデルが 512 トークンで打ち切ったときに残る範囲の文章を返す。

    トークナイザーと同じく NFKC で正規化してから、MeCab の語を順に WordPiece に分け、
    収まらなくなった語の手前で切る。語の途中では切らない。
    """
    text = unicodedata.normalize("NFKC", text)
    used = 0
    end = 0
    for word in tokenizer.word_tokenizer.tokenize(text):
        pieces = len(tokenizer.subword_tokenizer.tokenize(word))
        if used + pieces > BUDGET:
            break
        start = text.find(word, end)
        if start < 0:
            raise ValueError(
                f"トークナイザーの語が文章の中に見つかりません: {word!r}。"
                "トークナイザーの正規化の設定を確認してください。"
            )
        used += pieces
        end = start + len(word)
    return text[:end]


class _Feature(Protocol):
    pos1: str
    pos2: str
    pos3: str


class _Token(Protocol):
    surface: str
    feature: _Feature


class Tagger(Protocol):
    def __call__(self, text: str) -> Iterable[_Token]: ...


def new_tagger() -> Tagger:
    # fugashi は型情報を持たない拡張モジュールで、属性を型検査で引けないため、モジュールとして読む
    fugashi = importlib.import_module("fugashi")
    tagger: Tagger = fugashi.Tagger(f"-d {unidic_lite.DICDIR}")
    return tagger


def segment(text: str, tagger: Tagger) -> list[TaggedWord]:
    """記号・補助記号を除いた語を、出てきた順に返す。"""
    return [
        TaggedWord(
            word=Word(
                surface=token.surface,
                pos=token.feature.pos1,
                pos2=None if token.feature.pos2 == NO_SUBCLASS else token.feature.pos2,
            ),
            pos3=token.feature.pos3,
        )
        for token in tagger(text)
        if token.feature.pos1 not in EXCLUDED_POS
    ]
