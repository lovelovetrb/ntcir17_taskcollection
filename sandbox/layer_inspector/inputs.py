"""モデルに入力されたクエリと文書を、切り詰めたトークン列から取り出す。

元の文章ではなく、エンコードと同じく 512 トークンで切り詰めたトークン列を decode した
文字列を表示する。正規化で変わった文字、[UNK] になった文字、打ち切りで落ちた末尾が、
モデルが受け取ったとおりに見える (ADR-0020)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from hidden_subspace.corpus.documents import Document
from hidden_subspace.corpus.text import document_text, query_text
from hidden_subspace.corpus.topics import Topic
from hidden_subspace.encoding.encoder import MAX_TOKENS


@dataclass(frozen=True)
class ModelInputs:
    queries: dict[str, str]
    """トピック ID から、モデルに入力されたクエリへの対応。"""
    documents: dict[str, str]
    """文書 ID から、モデルに入力された文書への対応。"""


def decode_model_inputs(
    topics: Sequence[Topic],
    documents: Sequence[Document],
    tokenizer: PreTrainedTokenizerBase,
) -> ModelInputs:
    """特殊トークンは取り除かない。[UNK] を消すと、語彙にない文字があったことが見えなくなる。

    エンコードはバッチごとにパディングを付けて呼ぶが、パディングは平均から除かれるため、
    1 件ずつ呼んでもモデルが受け取るトークン列は変わらない。
    """
    return ModelInputs(
        queries={topic.topic_id: _decode(query_text(topic), tokenizer) for topic in topics},
        documents={
            document.document_id: _decode(document_text(document), tokenizer)
            for document in documents
        },
    )


def _decode(text: str, tokenizer: PreTrainedTokenizerBase) -> str:
    input_ids = tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    # decode の注釈は複数の列を渡した場合の list[str] を含むが、1 つの列なら str を返す
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    if not isinstance(decoded, str):
        raise TypeError(f"decode が文字列を返しませんでした: {type(decoded).__name__}")
    return decoded
