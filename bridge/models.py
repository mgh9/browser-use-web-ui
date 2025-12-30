"""Pydantic request/response models for the bridge API."""
from typing import Optional

from pydantic import BaseModel, Field


class RunTaskBody(BaseModel):
    task: str
    environment: str = Field(..., alias="Environment")
    startUrl: Optional[str] = None
    maxSteps: Optional[int] = None
    cdpUrl: Optional[str] = None
    userDataDir: Optional[str] = None
    clientTaskId: Optional[str] = None
    taskStartedCallbackUrl: Optional[str] = None
    taskCompletedCallbackUrl: Optional[str] = None
    llmProvider: Optional[str] = None
    llmModel: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
