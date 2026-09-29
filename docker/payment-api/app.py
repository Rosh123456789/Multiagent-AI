from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(title="Payment API")


@app.get("/health")
def health():
    mysql_host = os.getenv("MYSQL_HOST", "mysql")
    if mysql_host == "mysql":
        return {"status": "unhealthy", "message": "Database dependency unavailable."}
    return {"status": "healthy", "message": "Payment API healthy."}


@app.get("/")
def root():
    return {"service": "Payment API", "status": "ok"}
