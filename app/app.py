import sys
from pathlib import Path

# Ensure project root is importable when running Streamlit
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
import streamlit as st

from rag.rag_answer import answer_question_structured
from rag.index_store import load_index, load_metadata

# Professional defaults
DEFAULT_TOP_K = 5
DEFAULT_MEMORY_LEN = 2


@st.cache_resource(show_spinner=False)
def get_index():
    return load_index()


@st.cache_data(show_spinner=False)
def get_meta():
    return load_metadata()


INDEX = get_index()
META = get_meta()


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
        if y:
            years.add(y)

        for t in item.get("topics", []):
            topics.add(t)

    return {
        "topics": sorted(topics),
        "categories": sorted(categories),
        "years": sorted(years),
        "docs": sorted(docs),
    }


def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RAG Chatbot")
    st.caption("Grounded answers • OpenAI embeddings • FAISS vector search")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Filters")

        # Optional: cache reset button (useful after rebuilding index/meta)
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        opts = load_meta_options(META)

        category_ui = st.selectbox("Category", ["(any)"] + opts["categories"], 0)
        topics_ui = st.multiselect("Topics", opts["topics"])
        year_ui = st.selectbox("Year", ["(any)"] + [str(y) for y in opts["years"]], 0)
        doc_ui = st.selectbox("Document", ["(any)"] + opts["docs"], 0)

        # Convert UI selections -> filter args
        category_filter = None if category_ui == "(any)" else category_ui
        topic_filter = None if len(topics_ui) == 0 else topics_ui
        year_filter = None if year_ui == "(any)" else int(year_ui)
        doc_filter = None if doc_ui == "(any)" else [doc_ui]

    # ---- Session state ----
    if "messages" not in st.session_state:
        st.session_state.messages = []  # {"role":..., "content":..., "sources":..., "quotes":..., "confidence":...}

    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []

    # ---- Chat history render ----
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

            if m["role"] == "assistant":
                sources = m.get("sources", [])
                quotes = m.get("quotes", [])
                confidence = m.get("confidence")

                if confidence:
                    st.caption(f"Confidence: {confidence}")

                if sources:
                    with st.expander("Sources", expanded=False):
                        for s in sources:
                            st.write(f"- {s.get('doc')} (page {s.get('page')})")

                if quotes:
                    with st.expander("Supporting quotes", expanded=False):
                        for q in quotes:
                            quote = q.get("quote", "")
                            src_idx = q.get("source_index")
                            if src_idx is not None:
                                st.write(f'• "{quote}" (source #{src_idx})')
                            else:
                                st.write(f'• "{quote}"')

    # ---- Chat input ----
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

            st.markdown(result.get("answer", ""))

            sources = result.get("sources", [])
            quotes = result.get("quotes", [])
            confidence = result.get("confidence")

            if confidence:
                st.caption(f"Confidence: {confidence}")

            if sources:
                with st.expander("Sources", expanded=False):
                    for s in sources:
                        st.write(f"- {s.get('doc')} (page {s.get('page')})")

            if quotes:
                with st.expander("Supporting quotes", expanded=False):
                    for q in quotes:
                        quote = q.get("quote", "")
                        src_idx = q.get("source_index")
                        if src_idx is not None:
                            st.write(f'• "{quote}" (source #{src_idx})')
                        else:
                            st.write(f'• "{quote}"')

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "sources": result.get("sources", []),
                "quotes": result.get("quotes", []),
                "confidence": result.get("confidence"),
            }
        )


if __name__ == "__main__":
    main()




