import asyncio
from datetime import datetime
import os
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatBrowserUse
from browser_use.llm.openrouter.chat import ChatOpenRouter
from browser_use.llm.ollama.chat import ChatOllama
import logging
import httpx

app = FastAPI(title="Browser-Use Local Bridge")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

class RunTaskBody(BaseModel):
    task: str
    startUrl: Optional[str] = None
    maxSteps: Optional[int] = 30
    cdpUrl: Optional[str] = None
    userDataDir: Optional[str] = None
    clientTaskId: Optional[str] = None
    taskStartedCallbackUrl: Optional[str] = None
    taskCompletedCallbackUrl: Optional[str] = None


# simple in-memory task store; replace with Redis/DB if needed
TASKS: dict[str, dict] = {}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/run-task")
async def run_task(body: RunTaskBody):
    task_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    cdp_url = body.cdpUrl or os.getenv("BROWSER_CDP")
    logger.info(
        "[RUN_TASK] task_id=%s clientTaskId=%s cdpUrl=%s userDataDir=%s startUrl=%s maxSteps=%s",
        task_id,
        body.clientTaskId,
        cdp_url,
        body.userDataDir or os.getenv("BROWSER_USER_DATA"),
        body.startUrl,
        body.maxSteps,
    )
    TASKS[task_id] = {
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
        "taskStartedCallbackUrl": body.taskStartedCallbackUrl,
        "taskCompletedCallbackUrl": body.taskCompletedCallbackUrl,
        "clientTaskId": body.clientTaskId,
        "task": body.task,
        "startUrl": body.startUrl,
        "maxSteps": body.maxSteps,
    }
    handle: asyncio.Task | None = None
    # fire start callback (non-blocking)
    if body.taskStartedCallbackUrl:
        logger.info(
            "[START_CB] task_id=%s clientTaskId=%s url=%s",
            task_id,
            body.clientTaskId,
            body.taskStartedCallbackUrl,
        )
        asyncio.create_task(_send_callback(body.taskStartedCallbackUrl, task_id))

    async def runner():
        try:
            browser = Browser(
                cdp_url=cdp_url,
                user_data_dir=body.userDataDir or os.getenv("BROWSER_USER_DATA") or None,
            )
            llm = _get_llm()
            agent = Agent(
                task=body.task,
                browser=browser,
                llm=llm,
                start_url=body.startUrl,
                max_steps=body.maxSteps,
            )
            history = await agent.run()
            _mark_done(
                task_id,
                status="finished",
                is_success=True,
                steps=_serialize_history(history),
                output=_extract_output(history),
                error=None,
            )
        except asyncio.CancelledError:
            _mark_done(
                task_id,
                status="canceled",
                is_success=False,
                steps=[],
                output=None,
                error="canceled",
            )
            raise
        except Exception as exc:
            _mark_done(
                task_id,
                status="stopped",
                is_success=False,
                steps=[],
                output=None,
                error=str(exc),
            )

    handle = asyncio.create_task(runner())
    TASKS[task_id]["_handle"] = handle
    logger.info(
        "[SCHEDULED] task_id=%s clientTaskId=%s cdpUrl=%s",
        task_id,
        body.clientTaskId,
        cdp_url,
    )
    return {"id": task_id, "clientTaskId": body.clientTaskId}


@app.get("/task/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        return {"status": "not_found", "id": task_id}
    return {"id": task_id, **_public_task_view(TASKS[task_id])}


@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"id": task_id, "status": "not_found"}
    handle: asyncio.Task | None = task.get("_handle")
    if handle and not handle.done():
        handle.cancel()
        task["status"] = "canceled"
        logger.info(
            "[CANCEL] task_id=%s clientTaskId=%s",
            task_id,
            task.get("clientTaskId"),
        )
        return {"id": task_id, "clientTaskId": task.get("clientTaskId"), "status": "canceled"}
    return {"id": task_id, "clientTaskId": task.get("clientTaskId"), "status": task.get("status", "unknown")}


def _get_llm():
	"""
	Select LLM based on env vars.

	Priority:
	1) LLM_PROVIDER / LLM_MODEL
	2) DEFAULT_LLM / DEFAULT_MODEL_NAME (keeps parity with WebUI vars)
	Supported providers:
	- openrouter: OPENROUTER_API_KEY or OPENAI_API_KEY, OPENROUTER_BASE_URL or OPENAI_ENDPOINT
	- ollama: OLLAMA_ENDPOINT or OLLAMA_HOST
	- browser-use (default): needs BROWSER_USE_API_KEY
	"""
	provider = os.getenv("LLM_PROVIDER") or os.getenv("DEFAULT_LLM") or "browser-use"
	provider = provider.lower()
	model = os.getenv("LLM_MODEL") or os.getenv("DEFAULT_MODEL_NAME") or "bu-latest"

	if provider == "openrouter":
		api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
		base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_ENDPOINT") or "https://openrouter.ai/api/v1"
		logger.info(
			"LLM selection: provider=openrouter model=%s base_url=%s api_key_set=%s",
			model,
			base_url,
			bool(api_key),
		)
		return ChatOpenRouter(model=model, api_key=api_key, base_url=base_url)

	if provider == "ollama":
		host = os.getenv("OLLAMA_ENDPOINT") or os.getenv("OLLAMA_HOST")
		if not model:
			model = "llama3.2"
		logger.info(
			"LLM selection: provider=ollama model=%s host=%s host_set=%s",
			model,
			host,
			bool(host),
		)
		return ChatOllama(model=model, host=host)

	# default: Browser-Use cloud LLM (requires BROWSER_USE_API_KEY)
	bu_key = os.getenv("BROWSER_USE_API_KEY")
	bu_base = os.getenv("BROWSER_USE_LLM_URL")
	logger.info(
		"LLM selection: provider=browser-use model=%s base_url=%s api_key_set=%s",
		model,
		bu_base,
		bool(bu_key),
	)
	return ChatBrowserUse(model=model, api_key=bu_key, base_url=bu_base)


def _serialize_history(history):
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


def _extract_output(history) -> str:
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


def _public_task_view(task: dict) -> dict:
    """Strip internal fields (e.g., _handle) from task dict."""
    return {k: v for k, v in task.items() if not k.startswith("_")}


def _mark_done(task_id: str, status: str, is_success: bool, steps, output, error):
    finished_at = datetime.utcnow()
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
            "[COMPLETE_CB] task_id=%s clientTaskId=%s url=%s cdpUrl=%s",
            task_id,
            task.get("clientTaskId"),
            callback_url,
            task.get("cdpUrl"),
        )
        asyncio.create_task(_send_callback(callback_url, task_id))


async def _send_callback(callback_url: str, task_id: str):
    payload = {"id": task_id, **_public_task_view(TASKS.get(task_id, {}))}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(callback_url, json=payload)
    except Exception as exc:
        logger.warning("Callback failed for %s: %s", task_id, exc)
    else:
        logger.info("[CALLBACK_OK] task_id=%s url=%s", task_id, callback_url)
