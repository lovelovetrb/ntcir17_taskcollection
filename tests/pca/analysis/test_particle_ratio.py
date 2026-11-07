"""文書中の助詞・助動詞の割合を数える処理を確かめる。"""

from __future__ import annotations

import pytest

from pca.analysis.common.words import new_tagger, segment
from pca.analysis.particle_ratio.ratios import INDICATORS, ratios


def ratios_by_name(text: str) -> dict[str, float]:
    values = ratios(segment(text, new_tagger()))
    return dict(zip([indicator.name for indicator in INDICATORS], values, strict=True))


def test_ratios_are_counts_over_words() -> None:
    # 猫・が・犬・を・見・た の 6 語
    result = ratios_by_name("猫が犬を見た。")

    expected = {"particle": 2 / 6, "auxiliary": 1 / 6, "particle-ga": 1 / 6}
    expected |= {"particle-wo": 1 / 6, "auxiliary-ta": 1 / 6}
    assert result == pytest.approx({name: expected.get(name, 0.0) for name in result})


def test_same_surface_is_told_apart_by_part_of_speech() -> None:
    # 東京・で (格助詞)・会議・で (助動詞)・ある の 5 語
    result = ratios_by_name("東京で会議である。")

    assert result["particle-de"] == pytest.approx(1 / 5)
    assert result["auxiliary"] == pytest.approx(1 / 5)


def test_rejects_document_without_words() -> None:
    with pytest.raises(ValueError, match="語"):
        ratios([])
