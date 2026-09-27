"""PDF -> pages -> chunks -> embeddings -> Chroma upsert."""

from pypdf import PdfReader
import io

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text per page. Returns [{"filename", "page" (1-indexed), "text"}, ...]."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"filename": filename, "page": i + 1, "text": text})
    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window over paragraphs, falling back to a raw character window
    when a single paragraph exceeds chunk_size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself is too long: raw sliding window with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap
            current = ""

    if current:
        chunks.append(current)

    # Apply overlap between paragraph-based chunks by prefixing the tail of the
    # previous chunk, so context isn't lost at a boundary.
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            tail = chunks[i - 1][-overlap:]
            chunk = tail + "\n\n" + chunk
        overlapped.append(chunk)
    return overlapped


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk each page's text independently, so a chunk never crosses a page boundary."""
    chunks = []
    for page in pages:
        pieces = _split_text(page["text"], chunk_size, overlap)
        for idx, piece in enumerate(pieces):
            chunks.append({
                "filename": page["filename"],
                "page": page["page"],
                "chunk_index": idx,
                "text": piece,
            })
    return chunks


def embed_chunks(chunks: list[dict], embedder) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts).tolist()


def ingest_files(uploaded_files, collection, embedder) -> int:
    """Ingest a list of Streamlit UploadedFile objects into the given Chroma collection.
    Returns the number of chunks added."""
    total_chunks = 0
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        pages = extract_pages(file_bytes, uploaded_file.name)
        chunks = chunk_pages(pages)
        if not chunks:
            continue

        embeddings = embed_chunks(chunks, embedder)
        ids = [f"{c['filename']}::p{c['page']}::c{c['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"filename": c["filename"], "page": c["page"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ]

        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        total_chunks += len(chunks)

    return total_chunks
