"""文書ごとに、記号を除いた語に占める品詞と助詞・助動詞の割合を出す (ADR-0031、ADR-0033)。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import torch
from torch import Tensor
from transformers import BertJapaneseTokenizer

from pca.analysis.common.words import Tagger, Word, load_tokenizer, new_tagger, segment, truncate


@dataclass(frozen=True)
class Indicator:
    """数える語の条件。`surface` が None なら、品詞の大分類が `pos` の語をすべて数える。"""

    name: str
    label: str
    """図の横軸に書く英語の名前。"""
    pos: str
    surface: str | None = None
    pos2: str | None = None

    def matches(self, word: Word) -> bool:
        if word.pos != self.pos:
            return False
        return self.surface is None or (word.surface, word.pos2) == (self.surface, self.pos2)


INDICATORS = (
    Indicator("noun", "nouns", "名詞"),
    Indicator("particle", "particles", "助詞"),
    Indicator("verb", "verbs", "動詞"),
    Indicator("auxiliary", "auxiliary verbs", "助動詞"),
    Indicator("suffix", "suffixes", "接尾辞"),
    Indicator("prefix", "prefixes", "接頭辞"),
    Indicator("adjectival-noun", "adjectival nouns", "形状詞"),
    Indicator("particle-no", "no (case particle)", "助詞", "の", "格助詞"),
    Indicator("particle-wo", "wo (case particle)", "助詞", "を", "格助詞"),
    Indicator("particle-ni", "ni (case particle)", "助詞", "に", "格助詞"),
    Indicator("particle-te", "te (conjunctive particle)", "助詞", "て", "接続助詞"),
    Indicator("particle-wa", "wa (binding particle)", "助詞", "は", "係助詞"),
    Indicator("particle-to", "to (case particle)", "助詞", "と", "格助詞"),
    Indicator("particle-ga", "ga (case particle)", "助詞", "が", "格助詞"),
    Indicator("particle-de", "de (case particle)", "助詞", "で", "格助詞"),
    Indicator("auxiliary-ta", "ta (auxiliary verb)", "助動詞", "た"),
)


def ratios(words: Sequence[Word]) -> list[float]:
    """`INDICATORS` の順に、各条件に合う語の数を語の数で割った値を返す。"""
    if not words:
        raise ValueError("記号を除いた語がない文書があります。文書の本文を確認してください。")
    return [sum(indicator.matches(word) for word in words) / len(words) for indicator in INDICATORS]


_tokenizer: BertJapaneseTokenizer | None = None
_tagger: Tagger | None = None


def _ratios_of(model_id: str, text: str) -> list[float]:
    # Pool の initializer で読み込むと、失敗したときに Pool がワーカーを作り直し続けて
    # 終わらない。最初のタスクで読み込み、失敗を親プロセスに返す
    global _tokenizer, _tagger
    if _tokenizer is None or _tagger is None:
        _tokenizer = load_tokenizer(model_id)
        _tagger = new_tagger()
    return ratios(segment(truncate(text, _tokenizer, _tagger), _tagger))


def document_ratios(texts: Sequence[str], model_id: str, processes: int) -> Tensor:
    """`[文書数, 指標の数]` の割合を返す。"""
    with Pool(processes) as pool:
        return torch.tensor(
            pool.map(partial(_ratios_of, model_id), texts, chunksize=500), dtype=torch.float64
        )
