import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st

from rag.rag_answer import answer_question_structured
from rag.index_store import load_index, load_metadata

DEFAULT_TOP_K = 20
DEFAULT_MEMORY_LEN = 2


@st.cache_resource(show_spinner=False)
def get_index():
    return load_index()


@st.cache_data(show_spinner=False)
def get_meta():
    return load_metadata()


@st.cache_data(show_spinner=False)
def load_meta_options(meta: list[dict]):
    topics = set()
    categories = set()
    years = set()
    docs = set()

    for item in meta:
        d = item.get("doc")
        if d:
            docs.add(d)

        cat = item.get("category")
        if cat:
            categories.add(cat)

        y = item.get("year")
        if y is not None:
            years.add(y)

        for t in item.get("topics", []) or []:
            topics.add(t)

    return {
        "topics": sorted(topics),
        "categories": sorted(categories),
        "years": sorted(years),
        "docs": sorted(docs),
    }


def _filters_summary(doc_filter, year_filter, category_filter, topic_filter):
    parts = []
    if doc_filter:
        parts.append(f"Doc: {doc_filter[0]}")
    if year_filter is not None:
        parts.append(f"Year: {year_filter}")
    if category_filter:
        parts.append(f"Category: {category_filter}")
    if topic_filter:
        parts.append("Topics: " + ", ".join(topic_filter))
    return " • ".join(parts) if parts else "No filters applied."


def _dedupe_quotes(quotes: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for q in quotes or []:
        qt = (q.get("quote") or "").strip()
        if qt and qt not in seen:
            seen.add(qt)
            deduped.append(q)
    return deduped


def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RAG Chatbot")
    st.caption("Grounded answers • OpenAI embeddings • FAISS vector search")

    # Load data
    INDEX = get_index()
    META = get_meta()
    opts = load_meta_options(META)

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        # Clear filters
        if st.button("Clear filters", use_container_width=True):
            st.session_state["category_ui"] = "(any)"
            st.session_state["topics_ui"] = []
            st.session_state["year_ui"] = "(any)"
            st.session_state["doc_ui"] = "(any)"
            st.rerun()

        category_ui = st.selectbox(
            "Category",
            ["(any)"] + opts["categories"],
            index=0,
            key="category_ui",
        )

        topics_ui = st.multiselect(
            "Topics",
            opts["topics"],
            key="topics_ui",
        )

        year_ui = st.selectbox(
            "Year",
            ["(any)"] + [str(y) for y in opts["years"]],
            index=0,
            key="year_ui",
        )

        doc_ui = st.selectbox(
            "Document",
            ["(any)"] + opts["docs"],
            index=0,
            key="doc_ui",
        )

        category_filter = None if category_ui == "(any)" else category_ui
        topic_filter = None if len(topics_ui) == 0 else topics_ui
        year_filter = None if year_ui == "(any)" else int(year_ui)
        doc_filter = None if doc_ui == "(any)" else [doc_ui]

        st.divider()
        st.caption("Active filters")
        st.write(_filters_summary(doc_filter, year_filter, category_filter, topic_filter))

        st.divider()

        if st.button("Reload index/meta (after rebuild)", use_container_width=True):
            get_index.clear()
            get_meta.clear()
            load_meta_options.clear()
            st.rerun()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

            if m["role"] == "assistant":
                sources = m.get("sources", [])
                quotes = _dedupe_quotes(m.get("quotes", []))
                confidence = m.get("confidence")

                if confidence:
                    st.caption(f"Confidence: {confidence}")

                if sources:
                    with st.expander("Sources", expanded=False):
                        for i, s in enumerate(sources, start=1):
                            doc = s.get("doc", "Unknown doc")
                            page = s.get("page", "?")
                            score = s.get("score", None)
                            score_txt = f" • score: {score:.3f}" if isinstance(score, (int, float)) else ""
                            st.write(f"**{i}. {doc}** (page {page}){score_txt}")

                if quotes:
                    with st.expander("Supporting quotes", expanded=False):
                        quote_options = []
                        for q in quotes:
                            quote = (q.get("quote") or "").strip()
                            src_idx = q.get("source_index")
                            if quote:
                                if src_idx is not None:
                                    quote_options.append(f"[{src_idx}] {quote}")
                                else:
                                    quote_options.append(quote)

                        if quote_options:
                            selected = st.selectbox("Pick a quote", quote_options, index=0, key=f"quote_{id(m)}")
                            st.info(selected)
                        else:
                            st.caption("No quotes available.")

    # Chat input
    user_input = st.chat_input("Ask a question about the documents…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.user_questions.append(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        history = st.session_state.user_questions[:-1][-DEFAULT_MEMORY_LEN:]

        with st.chat_message("assistant"):
            with st.spinner("Retrieving…"):
                result = answer_question_structured(
                    user_input,
                    index=INDEX,
                    meta=META,
                    top_k=DEFAULT_TOP_K,
                    history=history,
                    doc_filter=doc_filter,
                    year_filter=year_filter,
                    category_filter=category_filter,
                    topic_filter=topic_filter,
                )

            answer = result.get("answer", "")
            st.markdown(answer)

            sources = result.get("sources", [])
            quotes = _dedupe_quotes(result.get("quotes", []))
            confidence = result.get("confidence")

            if confidence:
                st.caption(f"Confidence: {confidence}")

            if sources:
                with st.expander("Sources", expanded=False):
                    for i, s in enumerate(sources, start=1):
                        doc = s.get("doc", "Unknown doc")
                        page = s.get("page", "?")
                        score = s.get("score", None)
                        score_txt = f" • score: {score:.3f}" if isinstance(score, (int, float)) else ""
                        st.write(f"**{i}. {doc}** (page {page}){score_txt}")

            if quotes:
                with st.expander("Supporting quotes", expanded=False):
                    quote_options = []
                    for q in quotes:
                        quote = (q.get("quote") or "").strip()
                        src_idx = q.get("source_index")
                        if quote:
                            if src_idx is not None:
                                quote_options.append(f"[{src_idx}] {quote}")
                            else:
                                quote_options.append(quote)

                    if quote_options:
                        selected = st.selectbox("Pick a quote", quote_options, index=0, key="quote_live")
                        st.info(selected)
                    else:
                        st.caption("No quotes available.")

        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "quotes": quotes,
                "confidence": confidence,
            }
        )


if __name__ == "__main__":
    main()
