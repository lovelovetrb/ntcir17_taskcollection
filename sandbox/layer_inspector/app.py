"""層ごとの検索結果を読むビューア (ADR-0020)。

    make layer-inspector

結果ファイルがなければ、起動時にキャッシュを GPU に載せて作る。2 度目以降はファイルを読む
だけで、GPU もモデルも要らない。`streamlit run` はこのファイルのあるディレクトリしか import の
探索先に加えないため、`sandbox` を PYTHONPATH に入れて起動する。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import streamlit as st
import torch
from streamlit.elements.arrow import DataframeStateInput

from layer_inspector.documents import RankedDocument
from layer_inspector.inspection import (
    Inspection,
    InspectionSettings,
    build_inspection,
    load_or_build_inspection,
)
from layer_inspector.presentation import (
    document_rows,
    grade_label,
    layer_of,
    metric_rows,
    topic_rows,
)
from layer_inspector.selection import TopicChoice

SETTINGS = InspectionSettings(
    model_id="cl-tohoku/bert-base-japanese-v3",
    experiment="layer-sweep",
    metric="nDCG@1000",
    shallow_layers=(1, 2),
    deep_layers=(11, 12),
    topic_count=5,
    top_count=10,
    relevant_count=5,
)
COLLECTION_ROOT = (
    Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser() / "NTCIR-1"
)
CACHE_ROOT = Path("cache")
RESULTS_ROOT = Path("results")
RESULT_FILE = RESULTS_ROOT / SETTINGS.model_id / "layer-inspector" / "inspection.json"

DOCUMENT_HEIGHT = 640
TABLES = {
    "top": "検索結果の上位",
    "relevant_top": "適合文書の上位",
    "relevant_bottom": "適合文書の下位",
}


@st.cache_resource(
    show_spinner="結果ファイルを用意しています。初回はキャッシュの読み込みと検索し直しに数十秒かかります。"
)
def load() -> Inspection:
    # 記録と同じ順位にするには GPU で検索し直す。CPU では記録との照合で止まる。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_or_build_inspection(
        RESULT_FILE,
        SETTINGS,
        lambda: build_inspection(SETTINGS, COLLECTION_ROOT, CACHE_ROOT, RESULTS_ROOT, device),
    )


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


def show_topic_table(
    title: str, choices: Sequence[TopicChoice], inspection: Inspection, topic_id: str
) -> None:
    st.subheader(title)
    if not choices:
        st.caption("該当するトピックはありません。")
        return
    topic_ids = [choice.topic_id for choice in choices]
    key = f"{title}-{st.session_state['generation']}"
    st.dataframe(
        topic_rows(choices, inspection.inputs, SETTINGS.metric),
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
    documents: Sequence[RankedDocument],
    choice: TopicChoice,
    favors_shallow: bool,
    selected: tuple[str, int],
) -> None:
    st.markdown(f"**{TABLES[table]}**")
    if not documents:
        st.caption("該当する適合文書はありません。")
        return
    key = f"{table}-{st.session_state['generation']}"
    selected_table, selected_index = selected
    st.dataframe(
        document_rows(documents, favors_shallow, layer_of(choice.shallow), layer_of(choice.deep)),
        key=key,
        hide_index=True,
        on_select=partial(on_document_selected, key, table),
        selection_mode="single-row",
        selection_default=selected_row(selected_index if table == selected_table else None),
    )


def show_document(
    inspection: Inspection,
    choice: TopicChoice,
    favors_shallow: bool,
    document: RankedDocument,
    title: str,
) -> None:
    searched, other = (
        (layer_of(choice.shallow), layer_of(choice.deep))
        if favors_shallow
        else (layer_of(choice.deep), layer_of(choice.shallow))
    )
    searched_rank, other_rank = (
        (document.shallow_rank, document.deep_rank)
        if favors_shallow
        else (document.deep_rank, document.shallow_rank)
    )

    st.subheader(title)
    columns = st.columns(4)
    columns[0].metric(f"層 {searched} の順位", f"{searched_rank:,}")
    columns[1].metric(f"層 {other} の順位", f"{other_rank:,}")
    columns[2].metric("判定", grade_label(document.grade))
    columns[3].markdown(f"**文書 ID**  \n`{document.document_id}`")

    st.markdown(f"**クエリ** (トピック {choice.topic_id})")
    st.code(inspection.inputs.queries[choice.topic_id], language=None, wrap_lines=True)
    st.markdown("**文書**")
    st.code(
        inspection.inputs.documents[document.document_id],
        language=None,
        wrap_lines=True,
        height=DOCUMENT_HEIGHT,
    )


def main() -> None:
    st.set_page_config(page_title="Layer Inspector", layout="wide")
    inspection = load()
    selection = inspection.selection
    groups = {choice.topic_id: (choice, True) for choice in selection.shallow_favored} | {
        choice.topic_id: (choice, False) for choice in selection.deep_favored
    }
    if not groups:
        st.error("浅い層と深い層で差のあるトピックがありません。記録と設定を確認してください。")
        return

    st.session_state.setdefault("topic_id", next(iter(groups)))
    st.session_state.setdefault("document", ("top", 0))
    st.session_state.setdefault("generation", 0)

    topic_id: str = st.session_state["topic_id"]
    choice, favors_shallow = groups[topic_id]
    topic_documents = inspection.documents[topic_id]
    tables = {
        "top": topic_documents.top,
        "relevant_top": topic_documents.relevant_top,
        "relevant_bottom": topic_documents.relevant_bottom,
    }
    table, index = st.session_state["document"]
    if index >= len(tables[table]):
        table, index = "top", 0

    browse, reading = st.columns(2, gap="medium")
    with browse:
        topic_column, result_column = st.columns([0.9, 1.1], gap="small")
        with topic_column:
            show_topic_table("浅い層が優位", selection.shallow_favored, inspection, topic_id)
            show_topic_table("深い層が優位", selection.deep_favored, inspection, topic_id)
        with result_column:
            searched = layer_of(choice.shallow) if favors_shallow else layer_of(choice.deep)
            st.subheader(f"トピック {topic_id}")
            st.caption(f"層 {searched} の順位で並べています。")
            st.dataframe(metric_rows(choice), hide_index=True)
            for name, documents in tables.items():
                show_document_table(name, documents, choice, favors_shallow, (table, index))
    with reading:
        title = f"{TABLES[table]} {index + 1} 件目"
        show_document(inspection, choice, favors_shallow, tables[table][index], title)


main()
