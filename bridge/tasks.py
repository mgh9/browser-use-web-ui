"""In-memory task store, lifecycle updates, and callbacks."""
import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from utils import utc_now

logger = logging.getLogger(__name__)

# simple in-memory task store; replace with Redis/DB if needed
TASKS: dict[str, dict[str, Any]] = {}


def build_task_record(
    *,
    task_id: str,
    task: str,
    client_task_id: Optional[str],
    environment: str,
    cdp_url: Optional[str],
    start_url: Optional[str],
    max_steps: int,
    llm_meta: dict[str, str],
    started_at: datetime,
    task_started_callback_url: Optional[str],
    task_completed_callback_url: Optional[str],
) -> dict[str, Any]:
    return {
        "status": "running",
        "output": None,
        "isSuccess": None,
        "steps": [],
        "error": None,
        "sessionId": None,
        "startedAt": started_at.isoformat() + "Z",
        "finishedAt": None,
        "durationSec": None,
        "cdpUrl": cdp_url,
        "taskStartedCallbackUrl": task_started_callback_url,
        "taskCompletedCallbackUrl": task_completed_callback_url,
        "clientTaskId": client_task_id,
        "task": task,
        "startUrl": start_url,
        "maxSteps": max_steps,
        "llmProvider": llm_meta["provider"],
        "llmModel": llm_meta["model"],
        "environment": environment,
    }


def public_task_view(task: dict) -> dict:
    """Strip internal fields (e.g., _handle) from task dict."""
    public = {k: v for k, v in task.items() if not k.startswith("_")}
    env_val = public.get("environment")
    if env_val is not None and "Environment" not in public:
        public["Environment"] = env_val
    return public


def mark_done(task_id: str, status: str, is_success: bool, steps, output, error):
    finished_at = utc_now()
    task = TASKS.get(task_id)
    if not task:
        return
    started_at = task.get("startedAt")
    duration = None
    try:
        if started_at:
            # Normalize to naive UTC to avoid aware/naive subtraction issues
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00")).replace(tzinfo=None)
            duration = (finished_at - start_dt).total_seconds()
    except Exception:
        duration = None

    task["status"] = status
    task["isSuccess"] = is_success
    task["steps"] = steps
    task["output"] = output
    task["error"] = error
    task["finishedAt"] = finished_at.isoformat() + "Z"
    task["durationSec"] = duration

    callback_url = task.get("taskCompletedCallbackUrl")
    if callback_url:
        logger.info(
            "[COMPLETE_CB] task_id=%s clientTaskId=%s environment=%s url=%s cdpUrl=%s",
            task_id,
            task.get("clientTaskId"),
            task.get("environment"),
            callback_url,
            task.get("cdpUrl"),
        )
        asyncio.create_task(send_callback(callback_url, task_id))


async def send_callback(callback_url: str, task_id: str):
    payload = {"id": task_id, **public_task_view(TASKS.get(task_id, {}))}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(callback_url, json=payload)
    except Exception as exc:
        logger.warning("Callback failed for %s: %s", task_id, exc)
    else:
        logger.info("[CALLBACK_OK] task_id=%s url=%s", task_id, callback_url)
