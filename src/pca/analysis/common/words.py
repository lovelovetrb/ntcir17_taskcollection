"""モデルに入る範囲の文章を、MeCab (unidic-lite) で語に分ける (ADR-0031)。"""

from __future__ import annotations

import importlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import unidic_lite
from transformers import AutoTokenizer, BertJapaneseTokenizer

from hidden_subspace.encoding.encoder import MAX_TOKENS
from hidden_subspace.models import resolve_model

BUDGET = MAX_TOKENS - 2
"""[CLS] と [SEP] を除いて、文章のトークンに使える数。"""
EXCLUDED_POS = ("記号", "補助記号")
NO_SUBCLASS = "*"
"""中分類がないときに MeCab が返す値。"""


@dataclass(frozen=True)
class Word:
    """語を表層形・品詞の大分類・中分類の組で区別する。中分類がない品詞では `pos2` が None。"""

    surface: str
    pos: str
    """品詞の大分類 (名詞、助詞など)。"""
    pos2: str | None


def load_tokenizer(model_id: str) -> BertJapaneseTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(resolve_model(model_id))
    if not isinstance(tokenizer, BertJapaneseTokenizer):
        raise TypeError(
            f"{model_id} のトークナイザーが BertJapaneseTokenizer ではありません。"
            "MeCab で語に分けるモデルを指定してください。"
        )
    added = sorted(set(tokenizer.get_added_vocab()) - set(tokenizer.all_special_tokens))
    if added:
        raise ValueError(
            f"{model_id} のトークナイザーに追加トークン {added} があります。"
            "追加トークンを持たないトークナイザーを指定してください。"
        )
    return tokenizer


def truncate(text: str, tokenizer: BertJapaneseTokenizer, tagger: Tagger) -> str:
    """モデルが 512 トークンで打ち切ったときに残る範囲の文章を返す。

    トークナイザーと同じく NFKC で正規化する。トークンは `tokenizer.tokenize` で数え、
    特殊トークンや小文字化もモデルの入力と同じに扱う。切る位置は MeCab の語の切れ目から、
    収まるうちで最も後ろのものを二分探索で選ぶ。語の途中では切らない。

    正規化した文章に特殊トークン (`[MASK]` など) の文字列があれば止める。モデルは正規化の前に
    特殊トークンを判定するため数え方がずれ、内側で切ると後ろで切るほどトークンが多いとも
    限らなくなる。
    """
    text = unicodedata.normalize("NFKC", text)
    found = [token for token in tokenizer.all_special_tokens if token in text]
    if found:
        raise ValueError(
            f"文章に特殊トークン {found} の文字列があります。"
            "特殊トークンを含まない文章を渡してください。"
        )
    if _fits(text, tokenizer):
        return text
    ends = _word_ends(text, tagger)
    low, high = 0, len(ends)
    while low < high:
        middle = (low + high + 1) // 2
        if _fits(text[: ends[middle - 1]], tokenizer):
            low = middle
        else:
            high = middle - 1
    return text[: ends[low - 1]] if low else ""


def _fits(text: str, tokenizer: BertJapaneseTokenizer) -> bool:
    return len(tokenizer.tokenize(text)) <= BUDGET


def _word_ends(text: str, tagger: Tagger) -> list[int]:
    """MeCab の各語が終わる位置を、文章の先頭からの文字数で返す。"""
    ends = []
    end = 0
    for token in tagger(text):
        start = text.find(token.surface, end)
        if start < 0:
            raise ValueError(
                f"MeCab の語が文章の中に見つかりません: {token.surface!r}。"
                "MeCab の辞書の設定を確認してください。"
            )
        end = start + len(token.surface)
        ends.append(end)
    return ends


class _Feature(Protocol):
    pos1: str
    pos2: str


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


def segment(text: str, tagger: Tagger) -> list[Word]:
    """記号・補助記号を除いた語を、出てきた順に返す。"""
    return [
        Word(
            surface=token.surface,
            pos=token.feature.pos1,
            pos2=None if token.feature.pos2 == NO_SUBCLASS else token.feature.pos2,
        )
        for token in tagger(text)
        if token.feature.pos1 not in EXCLUDED_POS
    ]
