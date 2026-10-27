"""Local FastAPI process. Binds to 127.0.0.1 only. WebSocket lands in Phase 7."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tooclose.config import load_config, public_config

config = load_config()

app = FastAPI(
    title="TooClose",
    description="Local fingerprint-proximity warning. Video frames never reach this process.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Does not log request bodies."""
    return {"status": "ok", "service": "tooclose"}


@app.get("/config")
def get_config() -> dict:
    """Public subset of config.yaml for the frontend."""
    return public_config(config)
