import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ModelCandidate:
    id: str
    name: str
    provider: str
    intelligence_index: float | None
    coding_index: float | None
    blended_cost_per_1m: float | None
    input_cost_per_1m: float | None
    output_cost_per_1m: float | None
    output_tokens_per_second: float | None
    time_to_first_token_seconds: float | None
    release_date: date | None


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def normalize_model(raw: dict[str, Any]) -> ModelCandidate:
    evaluations = raw.get("evaluations") or {}
    pricing = raw.get("pricing") or {}
    creator = raw.get("model_creator") or {}

    return ModelCandidate(
        id=raw["id"],
        name=raw["name"],
        provider=creator.get("name", "Unknown"),
        intelligence_index=evaluations.get("artificial_analysis_intelligence_index"),
        coding_index=evaluations.get("artificial_analysis_coding_index"),
        blended_cost_per_1m=pricing.get("price_1m_blended_3_to_1"),
        input_cost_per_1m=pricing.get("price_1m_input_tokens"),
        output_cost_per_1m=pricing.get("price_1m_output_tokens"),
        output_tokens_per_second=raw.get("median_output_tokens_per_second"),
        time_to_first_token_seconds=raw.get("median_time_to_first_token_seconds"),
        release_date=parse_date(raw.get("release_date")),
    )


def normalize_models(data: dict[str, Any]) -> list[ModelCandidate]:
    return [normalize_model(item) for item in data.get("data", [])]


def get_models(
    fn: str = os.path.join(os.path.dirname(__file__), "artificial_analysis_llms.json"),
) -> list[ModelCandidate]:
    with open(fn, "r") as f:
        data = json.load(f)
    return normalize_models(data)
