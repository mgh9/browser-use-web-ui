import asyncio
import os
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from browser_use import Agent, Browser, ChatBrowserUse

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
            llm = ChatBrowserUse()
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
