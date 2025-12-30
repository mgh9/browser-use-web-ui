"""LLM selection and logging helpers."""
import logging
import os
from typing import Optional

from browser_use import ChatBrowserUse
from browser_use.llm.openrouter.chat import ChatOpenRouter
from browser_use.llm.ollama.chat import ChatOllama

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "bu-latest"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def format_llm_log(
    requested_provider: Optional[str],
    requested_model: Optional[str],
    llm_meta: dict,
) -> str:
    effective_combo = f"{llm_meta['provider']}:{llm_meta['model']}"
    requested_present = requested_provider or requested_model
    if not requested_present:
        return f"{effective_combo} (default)"
    requested_provider_display = requested_provider.strip() if isinstance(requested_provider, str) else None
    requested_model_display = requested_model.strip() if isinstance(requested_model, str) else None
    requested_provider_clean = requested_provider_display.lower() if requested_provider_display else None
    requested_model_clean = requested_model_display if requested_model_display else None
    if (
        (requested_provider_clean and requested_provider_clean != llm_meta["provider"])
        or (requested_model_clean and requested_model_clean != llm_meta["model"])
    ):
        req_disp = f"{requested_provider_display or '-'}:{requested_model_display or '-'}"
        return f"{effective_combo} (requested {req_disp})"
    return f"{effective_combo} (requested)"


def get_llm(override_provider: Optional[str] = None, override_model: Optional[str] = None):
    """
    Select LLM based on request overrides or env vars.

    Priority:
    1) per-request override (llmProvider / llmModel)
    2) LLM_PROVIDER / LLM_MODEL
    3) DEFAULT_LLM / DEFAULT_MODEL_NAME (keeps parity with WebUI vars)
    Supported providers:
    - openrouter: OPENROUTER_API_KEY or OPENAI_API_KEY, OPENROUTER_BASE_URL or OPENAI_ENDPOINT
    - ollama: OLLAMA_ENDPOINT or OLLAMA_HOST
    - browser-use (default): needs BROWSER_USE_API_KEY
    """
    provider_raw = override_provider or os.getenv("LLM_PROVIDER") or os.getenv("DEFAULT_LLM") or "browser-use"
    provider = provider_raw.strip().lower()
    model = (override_model or os.getenv("LLM_MODEL") or os.getenv("DEFAULT_MODEL_NAME") or DEFAULT_LLM_MODEL).strip()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_ENDPOINT") or DEFAULT_OPENROUTER_URL
        logger.info(
            "LLM selection: provider=openrouter model=%s base_url=%s api_key_set=%s",
            model,
            base_url,
            bool(api_key),
        )
        return ChatOpenRouter(model=model, api_key=api_key, base_url=base_url), {"provider": provider, "model": model}

    if provider == "ollama":
        host = os.getenv("OLLAMA_ENDPOINT") or os.getenv("OLLAMA_HOST")
        if not model:
            model = DEFAULT_OLLAMA_MODEL
        logger.info(
            "LLM selection: provider=ollama model=%s host=%s host_set=%s",
            model,
            host,
            bool(host),
        )
        return ChatOllama(model=model, host=host), {"provider": provider, "model": model}

    # default: Browser-Use cloud LLM (requires BROWSER_USE_API_KEY)
    bu_key = os.getenv("BROWSER_USE_API_KEY")
    bu_base = os.getenv("BROWSER_USE_LLM_URL")
    logger.info(
        "LLM selection: provider=browser-use model=%s base_url=%s api_key_set=%s",
        model,
        bu_base,
        bool(bu_key),
    )
    return ChatBrowserUse(model=model, api_key=bu_key, base_url=bu_base), {"provider": provider, "model": model}
