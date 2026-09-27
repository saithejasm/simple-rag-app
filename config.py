"""Central configuration: tunables and the provider table for the hybrid local/cloud LLM."""

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Local inference servers: reachable at a user-supplied URL, model list is
# discovered live by querying the server rather than hardcoded here.
LOCAL_SERVER_TYPES = {
    "Ollama": {
        "litellm_prefix": "ollama",
        "default_base_url": "http://localhost:11434",
        "models_path": "/api/tags",
    },
    "LM Studio / llama.cpp / vLLM (OpenAI-compatible)": {
        "litellm_prefix": "openai",
        "default_base_url": "http://localhost:1234/v1",
        "models_path": "/models",
    },
}

# Hosted cloud providers: fixed catalog, called through litellm as "<prefix>/<model>",
# require an API key.
CLOUD_PROVIDERS = {
    "OpenAI": {
        "prefix": "openai",
        "default_model": "gpt-4o-mini",
    },
    "Anthropic": {
        "prefix": "anthropic",
        "default_model": "claude-3-5-haiku-20241022",
    },
    "Google Gemini": {
        "prefix": "gemini",
        "default_model": "gemini-1.5-flash",
    },
    "xAI Grok": {
        "prefix": "xai",
        "default_model": "grok-2-latest",
    },
    "Groq": {
        "prefix": "groq",
        "default_model": "llama-3.1-8b-instant",
    },
    "MiniMax": {
        "prefix": "minimax",
        "default_model": "minimax-text-01",
    },
}
