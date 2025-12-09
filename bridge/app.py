import asyncio
import os
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatBrowserUse
from browser_use.llm.openrouter.chat import ChatOpenRouter
from browser_use.llm.ollama.chat import ChatOllama
import logging
import asyncio

app = FastAPI(title="Browser-Use Local Bridge")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

class RunTaskBody(BaseModel):
    task: str
    startUrl: Optional[str] = None
    maxSteps: Optional[int] = 30


# simple in-memory task store; replace with Redis/DB if needed
TASKS: dict[str, dict] = {}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/run-task")
async def run_task(body: RunTaskBody):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "running",
        "output": None,
        "isSuccess": None,
        "steps": [],
        "error": None,
        "sessionId": None,
    }
    handle: asyncio.Task | None = None

    async def runner():
        try:
            browser = Browser(
                cdp_url=os.getenv("BROWSER_CDP"),
                user_data_dir=os.getenv("BROWSER_USER_DATA") or None,
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
            TASKS[task_id]["status"] = "finished"
            TASKS[task_id]["isSuccess"] = True
            TASKS[task_id]["steps"] = _serialize_history(history)
            TASKS[task_id]["output"] = _extract_output(history)
        except asyncio.CancelledError:
            TASKS[task_id]["status"] = "canceled"
            TASKS[task_id]["isSuccess"] = False
            TASKS[task_id]["error"] = "canceled"
            raise
        except Exception as exc:
            TASKS[task_id]["status"] = "stopped"
            TASKS[task_id]["isSuccess"] = False
            TASKS[task_id]["error"] = str(exc)

    handle = asyncio.create_task(runner())
    TASKS[task_id]["_handle"] = handle
    return {"id": task_id}


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
        return {"id": task_id, "status": "canceled"}
    return {"id": task_id, "status": task.get("status", "unknown")}


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
