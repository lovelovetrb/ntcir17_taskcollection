"""層ごとの検索結果を読むビューア (ADR-0020, ADR-0021)。

    make layer-inspector

結果ファイルだけを読む。作るには `make layer-inspector-build` を使う。`streamlit run` は
このファイルのあるディレクトリしか import の探索先に加えないため、`sandbox` を PYTHONPATH に
入れて起動する。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import streamlit as st
from streamlit.elements.arrow import DataframeStateInput

from layer_inspector.presentation import document_rows, metric_rows, topic_rows
from layer_inspector.view import (
    DocumentView,
    InspectionView,
    TopicView,
    grade_label,
    read_view,
    readable,
)

RESULT_FILE = Path(
    os.environ.get(
        "LAYER_INSPECTOR_RESULT_FILE",
        "results/cl-tohoku/bert-base-japanese-v3/layer-inspector/inspection.json",
    )
)
DOCUMENT_HEIGHT = 640
TABLES = {
    "top": "検索結果の上位",
    "relevant_top": "適合文書の上位",
    "relevant_bottom": "適合文書の下位",
}


@st.cache_resource(show_spinner="結果ファイルを読んでいます。")
def load(path: Path) -> InspectionView:
    return read_view(path)


def refresh_tables() -> None:
    """表の選択はウィジェットごとに残る。キーを変えて作り直し、画面の選択を状態に揃える。"""
    st.session_state["generation"] += 1


def selected_row(index: int | None) -> DataframeStateInput | None:
    """作り直した表で、状態が選んでいる行を画面でも選んだ状態にする。"""
    return None if index is None else {"selection": {"rows": [index]}}


def on_topic_selected(key: str, topic_ids: Sequence[str]) -> None:
    rows = st.session_state[key]["selection"]["rows"]
    if rows:
        st.session_state["topic_id"] = topic_ids[rows[0]]
        st.session_state["document"] = ("top", 0)
    refresh_tables()


def on_document_selected(key: str, table: str) -> None:
    rows = st.session_state[key]["selection"]["rows"]
    if rows:
        st.session_state["document"] = (table, rows[0])
    refresh_tables()


def show_topic_table(title: str, topics: Sequence[TopicView], metric: str, topic_id: str) -> None:
    st.subheader(title)
    if not topics:
        st.caption("該当するトピックはありません。")
        return
    topic_ids = [topic.topic_id for topic in topics]
    key = f"{title}-{st.session_state['generation']}"
    st.dataframe(
        topic_rows(topics, metric),
        key=key,
        hide_index=True,
        on_select=partial(on_topic_selected, key, topic_ids),
        selection_mode="single-row",
        selection_default=selected_row(
            topic_ids.index(topic_id) if topic_id in topic_ids else None
        ),
    )


def show_document_table(
    table: str,
    documents: Sequence[DocumentView],
    topic: TopicView,
    selected: tuple[str, int],
) -> None:
    st.markdown(f"**{TABLES[table]}**")
    if not documents:
        st.caption("該当する適合文書はありません。")
        return
    key = f"{table}-{st.session_state['generation']}"
    selected_table, selected_index = selected
    st.dataframe(
        document_rows(documents, topic.favors_shallow, topic.shallow_layer, topic.deep_layer),
        key=key,
        hide_index=True,
        on_select=partial(on_document_selected, key, table),
        selection_mode="single-row",
        selection_default=selected_row(selected_index if table == selected_table else None),
    )


def show_document(
    view: InspectionView, topic: TopicView, document: DocumentView, title: str
) -> None:
    searched_rank, other_rank = (
        (document.shallow_rank, document.deep_rank)
        if topic.favors_shallow
        else (document.deep_rank, document.shallow_rank)
    )

    st.subheader(title)
    columns = st.columns(4)
    columns[0].metric(f"層 {topic.searched_layer} の順位", f"{searched_rank:,}")
    columns[1].metric(f"層 {topic.other_layer} の順位", f"{other_rank:,}")
    columns[2].metric("判定", grade_label(document.grade))
    columns[3].markdown(f"**文書 ID**  \n`{document.document_id}`")

    st.markdown(f"**クエリ** (トピック {topic.topic_id})")
    st.code(readable(topic.query), language=None, wrap_lines=True)
    st.markdown("**文書**")
    st.code(
        readable(view.documents[document.document_id]),
        language=None,
        wrap_lines=True,
        height=DOCUMENT_HEIGHT,
    )


def main() -> None:
    st.set_page_config(page_title="Layer Inspector", layout="wide")
    try:
        view = load(RESULT_FILE)
    except FileNotFoundError as error:
        st.error(str(error))
        return

    topics = {topic.topic_id: topic for topic in (*view.shallow_favored, *view.deep_favored)}
    if not topics:
        st.error("浅い層と深い層で差のあるトピックがありません。記録と設定を確認してください。")
        return

    st.session_state.setdefault("topic_id", next(iter(topics)))
    st.session_state.setdefault("document", ("top", 0))
    st.session_state.setdefault("generation", 0)

    topic = topics[st.session_state["topic_id"]]
    tables = {
        "top": topic.top,
        "relevant_top": topic.relevant_top,
        "relevant_bottom": topic.relevant_bottom,
    }
    table, index = st.session_state["document"]
    if index >= len(tables[table]):
        table, index = "top", 0

    browse, reading = st.columns(2, gap="medium")
    with browse:
        topic_column, result_column = st.columns([0.9, 1.1], gap="small")
        with topic_column:
            show_topic_table("浅い層が優位", view.shallow_favored, view.metric, topic.topic_id)
            show_topic_table("深い層が優位", view.deep_favored, view.metric, topic.topic_id)
        with result_column:
            st.subheader(f"トピック {topic.topic_id}")
            st.caption(f"層 {topic.searched_layer} の順位で並べています。")
            st.dataframe(metric_rows(topic), hide_index=True)
            for name, documents in tables.items():
                show_document_table(name, documents, topic, (table, index))
    with reading:
        title = f"{TABLES[table]} {index + 1} 件目"
        show_document(view, topic, tables[table][index], title)


main()
