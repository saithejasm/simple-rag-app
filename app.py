"""Streamlit UI: sidebar (provider/model/PDF upload) + chat loop."""

import time
import uuid

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CLOUD_PROVIDERS, EMBED_MODEL_NAME, LOCAL_SERVER_TYPES, TOP_K
from ingest import ingest_files
from introspect import build_tech_info, get_chunk_stats
from llm import build_model_string, fetch_local_models, generate_response
from retrieve import build_prompt, retrieve_chunks

st.set_page_config(page_title="Simple RAG app", page_icon="📄")


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=f"session_{st.session_state.session_id}")


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()
    if "local_models_cache" not in st.session_state:
        st.session_state.local_models_cache = {}
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


init_session_state()
embedder = get_embedder()
collection = get_collection()

# ---------------- Sidebar ----------------
SOURCE_OPTIONS = ["Local server"] + list(CLOUD_PROVIDERS.keys())

with st.sidebar:
    api_key = None
    api_base = None
    needs_key = False

    with st.expander("Settings", expanded=False):
        source = st.selectbox("LLM Source", SOURCE_OPTIONS, index=0)

        if source == "Local server":
            server_type = st.selectbox("Server type", list(LOCAL_SERVER_TYPES.keys()), index=0)
            server_config = LOCAL_SERVER_TYPES[server_type]
            prefix = server_config["litellm_prefix"]

            base_url = st.text_input(
                "Server URL",
                value=server_config["default_base_url"],
                key=f"base_url_{server_type}",
            )
            api_base = base_url

            cache_key = f"{server_type}|{base_url}"
            col_refresh, _ = st.columns([1, 2])
            refresh_clicked = col_refresh.button("Refresh models")

            if cache_key not in st.session_state.local_models_cache or refresh_clicked:
                try:
                    st.session_state.local_models_cache[cache_key] = fetch_local_models(server_type, base_url)
                except RuntimeError as e:
                    st.session_state.local_models_cache[cache_key] = []
                    st.error(str(e))

            available_models = st.session_state.local_models_cache.get(cache_key, [])
            if available_models:
                model_name = st.selectbox("Model", available_models)
            else:
                model_name = st.text_input("Model name (couldn't fetch list — enter manually)")

            api_key = st.text_input("API key (only if your server requires one)", type="password") or None
            if prefix == "openai" and not api_key:
                # litellm's OpenAI client requires a non-empty key even when the local
                # server (LM Studio, llama.cpp, vLLM) doesn't actually check it.
                api_key = "not-needed"
        else:
            prefix = CLOUD_PROVIDERS[source]["prefix"]
            needs_key = True
            api_key = st.text_input("API Key", type="password")
            model_name = st.text_input("Model name", value=CLOUD_PROVIDERS[source]["default_model"])

    model_string = build_model_string(prefix, model_name) if model_name else None

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_files]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} PDF(s)..."):
                n_chunks = ingest_files(new_files, collection, embedder)
            for f in new_files:
                st.session_state.ingested_files.add(f.name)
            st.success(f"Indexed {n_chunks} chunks from {len(new_files)} file(s).")

    if st.button("Clear all documents"):
        get_chroma_client().delete_collection(name=f"session_{st.session_state.session_id}")
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.ingested_files = set()
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

    st.caption(f"{collection.count()} chunks indexed from {len(st.session_state.ingested_files)} file(s).")

    # Rendered after ingestion above so live stats reflect this run's uploads, not the
    # previous rerun's state.
    with st.expander("How This Works", expanded=False):
        st.markdown(
            "**Pipeline:** PDF → `pypdf` per-page text extraction → paragraph-aware "
            "sliding-window chunking → local embeddings → ChromaDB → top-k retrieval → "
            "cited prompt → LLM via `litellm`. No LangChain."
        )
        stats = get_chunk_stats(collection)
        st.markdown(f"**Chunks indexed:** {stats['total_chunks']} total from {len(stats['files'])} file(s)")
        if stats["files"]:
            st.table([
                {"file": name, "pages": v["pages"], "chunks": v["chunks"]}
                for name, v in stats["files"].items()
            ])
        st.markdown(
            f"**Chunker:** paragraph-aware sliding window "
            f"(chunk_size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars, never crosses a page)"
        )
        st.markdown(
            f"**Embeddings:** `{EMBED_MODEL_NAME}` (sentence-transformers, "
            f"{embedder.get_embedding_dimension()} dimensions, computed locally)"
        )
        st.markdown(f"**Vector store:** ChromaDB, persistent, collection `session_{st.session_state.session_id}`")
        st.markdown(f"**Retrieval:** top-{TOP_K} chunks by embedding distance")
        st.markdown(f"**Current LLM:** {source} / {model_name or '(none selected)'} → `{model_string or '—'}`")
        st.caption("You can also just ask the chatbot these same questions directly.")

# ---------------- Main chat area ----------------
st.title("Simple RAG app")
st.caption("Built by [Sai Thejas Manjunath](https://saithejas.com)")
st.markdown(
    "Upload your own PDFs in the sidebar and ask questions about their content below. "
    "Curious how it works under the hood? Ask about the app itself too — e.g. \"how many "
    "chunks are indexed?\" or \"what chunker and embedding model do you use?\" — or expand "
    "**How This Works** in the sidebar and **Technical Details** under any answer to see "
    "exactly how retrieval and generation produced it."
)

def render_technical_details(details: dict):
    st.markdown(f"**Model used:** `{details['model_string']}`")
    st.markdown(
        f"**Latency:** retrieval {details['retrieval_ms']:.0f} ms, "
        f"generation {details['generation_ms']:.0f} ms"
    )
    if details["chunks"]:
        st.markdown("**Retrieved chunks (by distance, lower = more similar):**")
        st.table([
            {
                "filename": c["filename"],
                "page": c["page"],
                "distance": round(c["distance"], 4),
                "text": c["text"][:120] + ("..." if len(c["text"]) > 120 else ""),
            }
            for c in details["chunks"]
        ])
    st.markdown(
        f"**Query embedding:** {details['embedding_dim']} dimensions, first values: "
        f"`{[round(v, 4) for v in details['embedding_sample']]}`"
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for filename, page in message["sources"]:
                    st.markdown(f"- {filename}, page {page}")
        if message.get("technical_details"):
            with st.expander("Technical Details"):
                render_technical_details(message["technical_details"])

query = st.chat_input("Ask a question about your PDFs...")
if query:
    if collection.count() == 0:
        st.warning("Upload a PDF first.")
        st.stop()

    if not model_name:
        st.warning("Pick or enter a model name in the sidebar first.")
        st.stop()

    if needs_key and not api_key:
        st.warning(f"Enter an API key for {source}, or switch to a Local server.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieval_start = time.perf_counter()
            chunks, query_embedding = retrieve_chunks(query, collection, embedder)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

            tech_info = build_tech_info(collection, embedder, model_string, source, model_name)
            system_prompt, user_prompt = build_prompt(query, chunks, tech_info)

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_prompt}]

            generation_start = time.perf_counter()
            try:
                answer = generate_response(model_string, messages, api_key=api_key, api_base=api_base)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            generation_ms = (time.perf_counter() - generation_start) * 1000

        st.markdown(answer)
        sources = sorted({(c["filename"], c["page"]) for c in chunks})
        technical_details = {
            "model_string": model_string,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "chunks": chunks,
            "embedding_dim": len(query_embedding),
            "embedding_sample": query_embedding[:8],
        }
        if sources:
            with st.expander("Sources"):
                for filename, page in sources:
                    st.markdown(f"- {filename}, page {page}")
        with st.expander("Technical Details"):
            render_technical_details(technical_details)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "technical_details": technical_details,
    })
