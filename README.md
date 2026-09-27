# Simple RAG app

Built by [Sai Thejas Manjunath](https://saithejas.com)

A simple Retrieval-Augmented Generation (RAG) app: upload PDFs and chat about their
contents. Embeddings are always computed locally, and the chat model can be a free
local server (Ollama, LM Studio, llama.cpp, vLLM, ...) or any hosted cloud provider
you have an API key for. You can also query the app about its own technical
implementation — chunking, embeddings, chunk counts, current model, etc. — to see
how RAG works underneath.

```
PDF upload -> per-page text extraction -> chunking -> local embeddings -> ChromaDB
question -> embed -> retrieve top chunks -> prompt with citations -> litellm -> Ollama/cloud LLM
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A local inference server, for the free default option — any one of:
  - [Ollama](https://ollama.com): `brew install ollama && ollama serve && ollama pull llama3.2`
  - [LM Studio](https://lmstudio.ai): load a model and start its local server (Developer tab → Start Server)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) `server`, or vLLM — anything exposing an OpenAI-compatible `/v1/models` endpoint

## Setup

```
uv sync
uv run streamlit run app.py
```

## Usage

1. Upload one or more PDFs in the sidebar and wait for the "Indexed N chunks" message.
2. Ask questions in the chat box.
3. Expand "Sources" under each answer to see which file and page it came from.
4. Expand "Technical Details" under each answer to see the retrieved chunks and their
   similarity scores, the exact model that answered, retrieval/generation latency, and
   the query's embedding dimension.

## How this app is built (for demos / vivas)

Expand **"How This Works"** in the sidebar for a live-updating summary of the pipeline:
chunking method and parameters, embedding model, vector store, current chunk counts per
file, and the LLM currently in use.

You can also just **ask the chatbot directly** — questions like "how many chunks have
been indexed?", "what chunking method do you use?", or "which LLM are you?" are answered
from the app's own live state, in the same chat used for PDF questions.

## Choosing an LLM

In the sidebar, pick an **LLM Source**:

- **Local server**: pick a server type (Ollama, or LM Studio/llama.cpp/vLLM which
  all speak the OpenAI-compatible API), enter its URL (defaults are prefilled —
  `http://localhost:11434` for Ollama, `http://localhost:1234/v1` for the rest),
  and the app queries it live for the models it currently has available. Pick one
  from the dropdown. Click "Refresh models" after loading a different model on the
  server. Works with a server on another machine on your network too — just point
  the URL at it (e.g. `http://192.168.1.x:1234/v1`).
- **A hosted cloud provider** (OpenAI, Anthropic, Google Gemini, xAI Grok, Groq, or
  MiniMax): paste your API key and optionally adjust the model name. The key is
  kept only in memory for the session and is never written to disk.

## Note on hot-reload

`.streamlit/config.toml` disables Streamlit's file watcher. It isn't needed for the
app to work (reruns on user interaction happen regardless) — it only auto-reruns on
source save, and it's disabled here because it crashes when it scans `transformers`'
lazy-loaded module tree (needs `torchvision`, which this app doesn't use). After
editing code, just refresh the browser tab to pick up changes.

## Security note for public deployment

The "Local server" URL field makes the app's own server issue an HTTP request to
whatever URL is entered (to discover models and to run inference). That's fine when
you're the only one using the app on your own machine, but if this app is deployed
publicly, any visitor can enter an arbitrary URL and cause your server to make
outbound requests to it — a server-side request forgery (SSRF) risk, e.g. probing your
hosting provider's internal network or metadata endpoints. This is a known, accepted
limitation for now (not restricted in this version). If you deploy this publicly,
consider gating the "Local server" option behind an owner-only unlock, or removing it
and only offering hosted cloud providers.

## Known limitations

- Retrieval uses only the latest question, with no rewriting of follow-ups, so a
  question like "what about page 5?" that depends on earlier context may retrieve
  less relevant chunks.
- Only text-based PDFs are supported; scanned/image PDFs would need OCR (out of scope).
- Documents are stored per browser session; restarting the app starts with an empty
  document store.

## Project structure

- `app.py` — Streamlit UI: sidebar, chat loop, orchestrates the modules below.
- `ingest.py` — PDF → pages → chunks → embeddings → Chroma upsert.
- `retrieve.py` — Chroma query + prompt assembly with source citations; routes each
  question to a content-answering or app-introspection prompt.
- `introspect.py` — live chunk/file stats and the "about this app" facts used both by
  the sidebar panel and by the chatbot's self-introspection answers.
- `llm.py` — litellm wrapper: local server model discovery + generation for local/cloud.
- `config.py` — tunables, local server types, and the cloud provider table.
