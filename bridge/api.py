"""HTTP endpoints for task lifecycle (run, status, cancel)."""
import asyncio
import logging
import uuid

from fastapi import APIRouter

from browser_use import Agent, Browser

from llm import format_llm_log, get_llm
from models import RunTaskBody
from tasks import TASKS, build_task_record, mark_done, public_task_view, send_callback
from utils import (
    effective_max_steps,
    extract_output,
    format_max_steps_log,
    normalize_environment,
    read_positive_int_env,
    resolve_cdp_url,
    resolve_user_data_dir,
    serialize_history,
    utc_now,
)

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS_FALLBACK = 30
DEFAULT_MAX_STEPS_ENV = "DEFAULT_MAX_STEPS"
MAX_STEPS_LIMIT_ENV = "MAX_STEPS_LIMIT"


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.post("/run-task")
async def run_task(body: RunTaskBody):
    task_id = str(uuid.uuid4())
    started_at = utc_now()
    environment = normalize_environment(body.environment)
    cdp_url = resolve_cdp_url(body.cdpUrl)
    default_max_steps = read_positive_int_env(DEFAULT_MAX_STEPS_ENV, DEFAULT_MAX_STEPS_FALLBACK)
    max_steps_limit = read_positive_int_env(MAX_STEPS_LIMIT_ENV, default_max_steps)
    max_steps = effective_max_steps(body.maxSteps, default_max_steps, max_steps_limit)
    llm, llm_meta = get_llm(body.llmProvider, body.llmModel)
    max_steps_log = format_max_steps_log(body.maxSteps, max_steps, default_max_steps, max_steps_limit)
    llm_log = format_llm_log(body.llmProvider, body.llmModel, llm_meta)
    user_data_dir = resolve_user_data_dir(body.userDataDir)
    logger.info(
        "[RUN_TASK] task_id=%s clientTaskId=%s environment=%s cdpUrl=%s userDataDir=%s startUrl=%s maxSteps=%s llm=%s",
        task_id,
        body.clientTaskId,
        environment,
        cdp_url,
        user_data_dir,
        body.startUrl,
        max_steps_log,
        llm_log,
    )
    TASKS[task_id] = build_task_record(
        task_id=task_id,
        task=body.task,
        client_task_id=body.clientTaskId,
        environment=environment,
        cdp_url=cdp_url,
        start_url=body.startUrl,
        max_steps=max_steps,
        llm_meta=llm_meta,
        started_at=started_at,
        task_started_callback_url=body.taskStartedCallbackUrl,
        task_completed_callback_url=body.taskCompletedCallbackUrl,
    )
    # fire start callback (non-blocking)
    if body.taskStartedCallbackUrl:
        logger.info(
            "[START_CB] task_id=%s clientTaskId=%s environment=%s url=%s",
            task_id,
            body.clientTaskId,
            environment,
            body.taskStartedCallbackUrl,
        )
        asyncio.create_task(send_callback(body.taskStartedCallbackUrl, task_id))

    async def runner():
        try:
            browser = Browser(
                cdp_url=cdp_url,
                user_data_dir=user_data_dir,
            )
            agent = Agent(
                task=body.task,
                browser=browser,
                llm=llm,
                start_url=body.startUrl,
                max_steps=max_steps,
            )
            history = await agent.run(max_steps=max_steps)
            mark_done(
                task_id,
                status="finished",
                is_success=True,
                steps=serialize_history(history),
                output=extract_output(history),
                error=None,
            )
        except asyncio.CancelledError:
            mark_done(
                task_id,
                status="canceled",
                is_success=False,
                steps=[],
                output=None,
                error="canceled",
            )
            raise
        except Exception as exc:
            logger.exception("[RUN_TASK_ERROR] task_id=%s clientTaskId=%s", task_id, body.clientTaskId)
            mark_done(
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
        "[SCHEDULED] task_id=%s clientTaskId=%s environment=%s cdpUrl=%s",
        task_id,
        body.clientTaskId,
        environment,
        cdp_url,
    )
    return {"id": task_id, "clientTaskId": body.clientTaskId, "environment": environment}


@router.get("/task/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        return {"status": "not_found", "id": task_id}
    return {"id": task_id, **public_task_view(TASKS[task_id])}


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"id": task_id, "status": "not_found"}
    handle: asyncio.Task | None = task.get("_handle")
    if handle and not handle.done():
        handle.cancel()
        task["status"] = "canceled"
        logger.info(
            "[CANCEL] task_id=%s clientTaskId=%s environment=%s",
            task_id,
            task.get("clientTaskId"),
            task.get("environment"),
        )
        return {"id": task_id, "clientTaskId": task.get("clientTaskId"), "status": "canceled"}
    return {"id": task_id, "clientTaskId": task.get("clientTaskId"), "status": task.get("status", "unknown")}
