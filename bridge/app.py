import asyncio
import os
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatBrowserUse
from browser_use.llm.openrouter.chat import ChatOpenRouter
from browser_use.llm.ollama.chat import ChatOllama

app = FastAPI(title="Browser-Use Local Bridge")

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

    async def runner():
        try:
            browser = Browser(cdp_url=os.getenv("BROWSER_CDP"))
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
            TASKS[task_id]["steps"] = history
            TASKS[task_id]["output"] = history[-1]["content"] if history else ""
        except Exception as exc:
            TASKS[task_id]["status"] = "stopped"
            TASKS[task_id]["isSuccess"] = False
            TASKS[task_id]["error"] = str(exc)

    asyncio.create_task(runner())
    return {"id": task_id}


@app.get("/task/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        return {"status": "not_found", "id": task_id}
    return {"id": task_id, **TASKS[task_id]}


def _get_llm():
    """
    Select LLM based on env vars.

    Supported:
    - LLM_PROVIDER=openrouter with OPENROUTER_API_KEY/OPENAI_API_KEY and LLM_MODEL
    - LLM_PROVIDER=ollama with OLLAMA_ENDPOINT/OLLAMA_HOST and LLM_MODEL
    - default: ChatBrowserUse (requires BROWSER_USE_API_KEY)
    """
    provider = os.getenv("LLM_PROVIDER", "browser-use").lower()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenRouter(model=model, api_key=api_key, base_url=base_url)

    if provider == "ollama":
        host = os.getenv("OLLAMA_ENDPOINT") or os.getenv("OLLAMA_HOST")
        model = os.getenv("LLM_MODEL", "llama3.2")
        return ChatOllama(model=model, host=host)

    # default: Browser-Use cloud LLM (requires BROWSER_USE_API_KEY)
    model = os.getenv("LLM_MODEL", "bu-latest")
    return ChatBrowserUse(model=model, api_key=os.getenv("BROWSER_USE_API_KEY"), base_url=os.getenv("BROWSER_USE_LLM_URL"))
