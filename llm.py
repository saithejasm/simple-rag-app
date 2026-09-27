"""litellm wrapper: discovers models from a local inference server (Ollama,
LM Studio, llama.cpp, vLLM, ...) or calls a hosted cloud provider, all through
the same litellm.completion() call."""

import requests
import litellm
from litellm.exceptions import APIConnectionError, AuthenticationError

from config import LOCAL_SERVER_TYPES

REQUEST_TIMEOUT = 5


def fetch_local_models(server_type: str, base_url: str) -> list[str]:
    """Query a local inference server's URL for the models it currently has loaded/available."""
    base_url = base_url.rstrip("/")
    server = LOCAL_SERVER_TYPES[server_type]
    url = base_url + server["models_path"]

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach {base_url}. Is the server running? ({e})") from e
    except ValueError as e:
        raise RuntimeError(f"{base_url} did not return valid JSON from {server['models_path']}.") from e

    if server["litellm_prefix"] == "ollama":
        return sorted(m["name"] for m in data.get("models", []))
    return sorted(m["id"] for m in data.get("data", []))


def build_model_string(prefix: str, model_name: str) -> str:
    return f"{prefix}/{model_name}"


def generate_response(
    model_string: str,
    messages: list[dict],
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    kwargs = {"model": model_string, "messages": messages}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
    except APIConnectionError as e:
        target = api_base or model_string.split("/")[0]
        raise RuntimeError(f"Could not reach {target}. Is the server running? ({e})") from e
    except AuthenticationError as e:
        raise RuntimeError(f"Invalid API key: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e
