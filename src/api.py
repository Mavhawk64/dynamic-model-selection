import asyncio
import dataclasses
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

try:
    from src.artificial_analysis.catalog import ModelCandidate, get_models, normalize_models
    from src.artificial_analysis.client import load_or_fetch_llm_models
    from src.artificial_analysis.resolver import resolve
    from src.routing_types import SelectionPolicy
    from src.text_model_selector import select_text_model
except ImportError:
    from artificial_analysis.catalog import ModelCandidate, get_models, normalize_models
    from artificial_analysis.client import load_or_fetch_llm_models
    from artificial_analysis.resolver import resolve
    from routing_types import SelectionPolicy
    from text_model_selector import select_text_model


_candidates: list[ModelCandidate] = []

MODEL_REFRESH_INTERVAL_SECONDS = 60 * 60 * 24  # 24 hours

logger = logging.getLogger(__name__)


async def _refresh_models_loop() -> None:
    global _candidates
    while True:
        await asyncio.sleep(MODEL_REFRESH_INTERVAL_SECONDS)
        try:
            data = await asyncio.to_thread(load_or_fetch_llm_models, force_refresh=True)
            _candidates = normalize_models(data)
            logger.info("Model catalog refreshed: %d models loaded.", len(_candidates))
        except Exception as exc:
            logger.warning("Model catalog refresh failed, keeping existing models: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _candidates
    try:
        data = await asyncio.to_thread(load_or_fetch_llm_models, force_refresh=True)
        _candidates = normalize_models(data)
        logger.info("Loaded %d models from live AA API.", len(_candidates))
    except Exception as exc:
        logger.warning("Live AA fetch failed at startup, falling back to cache: %s", exc)
        _candidates = get_models()
    task = asyncio.create_task(_refresh_models_loop())
    yield
    task.cancel()


app = FastAPI(
    title="Dynamic Model Selector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mavhawk64.github.io",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SelectRequest(BaseModel):
    query: str
    policy: str = "nopref"
    big3_only: bool = True

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        valid = {p.value for p in SelectionPolicy}
        if v not in valid:
            raise ValueError(f"policy must be one of {sorted(valid)}")
        return v


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="<html><head><meta http-equiv='refresh' content='0; URL=/docs'></head></html>", status_code=200)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "models_loaded": str(len(_candidates))}


@app.post("/select")
def select(req: SelectRequest) -> dict[str, Any]:
    policy = SelectionPolicy(req.policy)
    selection = select_text_model(req.query, selection_policy=policy)
    result = resolve(selection, candidates=_candidates, big3_only=req.big3_only)

    return {
        "selection": dataclasses.asdict(selection),
        "result": json.loads(result.to_json()),
    }
