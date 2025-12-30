"""FastAPI app entrypoint; wires the API router and logging."""
import logging
import os

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

from fastapi import FastAPI

import api

app = FastAPI(title="Vendoroo Browser-Use Local Bridge")
app.include_router(api.router)
