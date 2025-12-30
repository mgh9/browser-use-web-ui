"""Shared utility functions for env parsing and history/output handling."""
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.utcnow()


def normalize_environment(environment: Optional[str]) -> str:
    if isinstance(environment, str):
        return environment.strip()
    return str(environment)


def resolve_cdp_url(cdp_url: Optional[str]) -> Optional[str]:
    return cdp_url or os.getenv("BROWSER_CDP")


def resolve_user_data_dir(user_data_dir: Optional[str]) -> Optional[str]:
    return user_data_dir or os.getenv("BROWSER_USER_DATA") or None


def effective_max_steps(requested: Optional[int], default_max: int, limit: int) -> int:
    effective = default_max
    if isinstance(requested, int) and requested > 0:
        effective = requested
    effective = min(effective, limit)
    return effective


def read_positive_int_env(var_name: str, fallback: int) -> int:
    value = os.getenv(var_name)
    if value is None:
        return fallback
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning("%s must be a positive integer; using %s", var_name, fallback)
        return fallback


def format_max_steps_log(
    requested: Optional[int], effective: int, default_max: int, limit: int
) -> str:
    requested_valid = isinstance(requested, int) and requested > 0
    if requested_valid:
        if requested > limit:
            return f"{effective} (requested {requested}, capped at limit {limit})"
        return f"{effective} (requested)"
    if requested is not None and not requested_valid:
        return f"{effective} (invalid request {requested}, using default {default_max})"
    if default_max == limit:
        return f"{effective} (default)"
    return f"{effective} (default {default_max}, limit {limit})"


def serialize_history(history):
    """Convert AgentHistoryList or list-like to JSON-serializable structure."""
    if hasattr(history, "model_dump"):
        try:
            return history.model_dump()
        except Exception:
            pass
    try:
        return list(history)
    except Exception:
        return str(history)


def extract_output(history) -> str:
    """
    Best-effort extraction of final output from AgentHistoryList.
    Returns empty string if not found.
    """
    try:
        hist_list = history.history if hasattr(history, "history") else history
        if not hist_list:
            return ""
        last_item = hist_list[-1]
        # pydantic model: access attributes
        if hasattr(last_item, "result"):
            res = last_item.result
            if res:
                last_res = res[-1]
                if hasattr(last_res, "extracted_content"):
                    return last_res.extracted_content or ""
                if hasattr(last_res, "model_dump"):
                    dumped = last_res.model_dump()
                    if isinstance(dumped, dict):
                        return dumped.get("extracted_content", "") or dumped.get("output", "") or ""
                    return str(dumped)
        # dict-like
        if isinstance(last_item, dict):
            results = last_item.get("result") or last_item.get("results")
            if results:
                last_res = results[-1]
                if isinstance(last_res, dict):
                    return last_res.get("extracted_content", "") or last_res.get("output", "") or ""
        return ""
    except Exception:
        return ""
