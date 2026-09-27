"""Chroma retrieval + prompt assembly with source citations."""

from config import TOP_K

# Small local models get noticeably less reliable at answering from document context when
# the system prompt also carries unrelated "about this app" facts (empirically: ~100% -> ~33%
# correct on a fixed test question with llama3.2:3b). So the two prompts are kept separate,
# and a cheap keyword check routes each question to the one it actually needs.
CONTENT_SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context excerpts
from uploaded PDF documents. Each excerpt is labeled with its source filename and page number.

Rules:
- Answer using only the information in the context below. If the answer is not in the
  context, say "I couldn't find this in the uploaded documents."
- When you use information from an excerpt, cite it inline like (filename, p. X).
- If multiple documents are relevant, mention which document each piece of information
  came from."""

META_SYSTEM_PROMPT_TEMPLATE = """You are the assistant for a RAG PDF chat app. The user is asking about how
this app itself is built or its current state, not about the uploaded PDFs' content. Answer
directly and confidently using only these facts:

{tech_info}"""

META_KEYWORDS = [
    "chunk", "embed", "vector store", "chromadb", "chroma", "litellm", " rag ", "retrieval",
    "top-k", "top k", "overlap", "architecture", "tech stack", "how were you built",
    "how was this built", "how was this app built", "how does this app work",
    "how do you work", "which llm", "which model", "what model are you", "what llm are you",
    "how many files", "how many documents", "how many pdfs",
]


def is_meta_question(query: str) -> bool:
    """Cheap keyword heuristic: is this question about the app's own implementation
    rather than about the uploaded PDFs' content?"""
    q = f" {query.lower()} "
    return any(kw in q for kw in META_KEYWORDS)


def retrieve_chunks(query: str, collection, embedder, top_k: int = TOP_K) -> tuple[list[dict], list[float]]:
    """Embed the query and return (top_k most relevant chunks with metadata, query embedding)."""
    query_embedding = embedder.encode([query]).tolist()[0]

    if collection.count() == 0:
        return [], query_embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "filename": meta["filename"],
            "page": meta["page"],
            "distance": distance,
        })
    return chunks, query_embedding


def build_prompt(query: str, chunks: list[dict], tech_info: str) -> tuple[str, str]:
    """Assemble the system prompt and a user prompt with cited context. Routes to the
    meta-question prompt (using tech_info) or the content-question prompt, whichever
    the query actually needs, to keep small local models reliable on both."""
    if is_meta_question(query):
        system_prompt = META_SYSTEM_PROMPT_TEMPLATE.format(tech_info=tech_info)
    else:
        system_prompt = CONTENT_SYSTEM_PROMPT

    if not chunks:
        context_block = "(no relevant excerpts found)"
    else:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f'[{i}] ({chunk["filename"]}, page {chunk["page"]}): "{chunk["text"]}"')
        context_block = "\n\n".join(lines)

    user_prompt = f"Document context:\n{context_block}\n\nQuestion: {query}"
    return system_prompt, user_prompt
