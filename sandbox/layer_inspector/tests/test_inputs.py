"""モデルに入力されたクエリと文書を、切り詰めたトークン列から取り出す処理を確かめる。

実際のモデルのトークナイザは読み込まず、小さな語彙から作ったトークナイザで振る舞いを見る。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import BertTokenizer

from hidden_subspace.corpus.documents import Document
from hidden_subspace.corpus.topics import Topic
from hidden_subspace.encoding.encoder import MAX_TOKENS
from layer_inspector.inputs import decode_model_inputs

VOCABULARY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "title", "abstract", "query", "word"]


@pytest.fixture
def tokenizer(tmp_path: Path) -> BertTokenizer:
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("\n".join(VOCABULARY), encoding="utf-8")
    return BertTokenizer(str(vocabulary))


def test_a_long_document_is_cut_to_the_model_limit(tokenizer: BertTokenizer) -> None:
    """モデルは 512 トークンまでしか受け取らない (ADR-0004)。落ちた末尾は表示しない。"""
    document = Document("d1", "title", "word " * 600)

    inputs = decode_model_inputs([], [document], tokenizer)

    tokens = inputs.documents["d1"].split()
    assert len(tokens) == MAX_TOKENS
    assert tokens[0] == "[CLS]"
    assert tokens[-1] == "[SEP]"


def test_special_tokens_are_kept(tokenizer: BertTokenizer) -> None:
    """[UNK] を取り除くと、語彙にない語があったこと自体が見えなくなる。"""
    document = Document("d1", "title", "abstract unseen")

    inputs = decode_model_inputs([], [document], tokenizer)

    assert inputs.documents["d1"] == "[CLS] title abstract [UNK] [SEP]"


def test_the_query_is_the_topic_title_alone(tokenizer: BertTokenizer) -> None:
    """検索には題名だけを使っており (ADR-0011)、DESCRIPTION はモデルに渡っていない。"""
    topic = Topic("0001", title="query", description="word word")

    inputs = decode_model_inputs([topic], [], tokenizer)

    assert inputs.queries["0001"] == "[CLS] query [SEP]"
