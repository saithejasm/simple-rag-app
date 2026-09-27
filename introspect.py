"""Live introspection: lets the sidebar panel and the chat itself report accurate,
up-to-date facts about how this RAG app is built and what state it's currently in."""

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL_NAME, TOP_K


def get_chunk_stats(collection) -> dict:
    """Returns {"total_chunks": int, "files": {filename: {"chunks": int, "pages": int}}}."""
    if collection.count() == 0:
        return {"total_chunks": 0, "files": {}}

    data = collection.get(include=["metadatas"])
    files: dict[str, dict] = {}
    for meta in data["metadatas"]:
        entry = files.setdefault(meta["filename"], {"chunks": 0, "pages": set()})
        entry["chunks"] += 1
        entry["pages"].add(meta["page"])

    return {
        "total_chunks": collection.count(),
        "files": {name: {"chunks": v["chunks"], "pages": len(v["pages"])} for name, v in files.items()},
    }


def build_tech_info(collection, embedder, model_string: str, source_label: str, model_name: str) -> str:
    """A live 'about this app' fact block, injected into the system prompt every turn so
    the chatbot can accurately answer meta-questions about its own implementation."""
    stats = get_chunk_stats(collection)
    embed_dim = embedder.get_embedding_dimension()

    if stats["files"]:
        file_lines = "\n".join(
            f"  - {name}: {v['chunks']} chunks, {v['pages']} pages"
            for name, v in stats["files"].items()
        )
    else:
        file_lines = "  (no documents indexed yet)"

    return f"""About this app (answer questions about the app's own implementation using these facts):
- Pipeline: PDF upload -> pypdf per-page text extraction -> paragraph-aware sliding-window
  chunking (chunk_size={CHUNK_SIZE} characters, overlap={CHUNK_OVERLAP} characters, a chunk
  never crosses a page boundary) -> local embeddings (sentence-transformers, "{EMBED_MODEL_NAME}",
  {embed_dim} dimensions, no API calls) -> ChromaDB (persistent, local vector store) -> top-{TOP_K}
  retrieval by embedding distance -> a prompt with cited context -> an LLM called via litellm.
- No LangChain is used anywhere; the whole pipeline is plain Python.
- Currently indexed: {stats['total_chunks']} chunks from {len(stats['files'])} file(s):
{file_lines}
- Currently answering with: {source_label} / {model_name} (litellm model string: "{model_string}")."""
