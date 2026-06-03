import json
import os
from pathlib import Path
from typing import Any

import requests

AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CLIENT_PATH = Path(__file__).parent
DEFAULT_CACHE_PATH = CLIENT_PATH / "artificial_analysis_llms.json"


def fetch_llm_models(api_key: str) -> dict[str, Any]:
    response = requests.get(
        AA_MODELS_URL,
        headers={"x-api-key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def load_or_fetch_llm_models(
    *,
    api_key: str | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
) -> dict[str, Any]:
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())

    api_key = api_key or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")

    if not api_key:
        raise RuntimeError("Missing ARTIFICIAL_ANALYSIS_API_KEY environment variable.")

    data = fetch_llm_models(api_key)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    return data


def extract_model_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("data", [])


if __name__ == "__main__":
    data = load_or_fetch_llm_models(force_refresh=True)
    models = extract_model_rows(data)

    print(f"Fetched {len(models)} models")

    for model in models[:10]:
        evaluations = model.get("evaluations", {})
        pricing = model.get("pricing", {})
        creator = model.get("model_creator", {})

        print(
            model.get("name"),
            "|",
            creator.get("name"),
            "| intelligence:",
            evaluations.get("artificial_analysis_intelligence_index"),
            "| coding:",
            evaluations.get("artificial_analysis_coding_index"),
            "| blended $/1M:",
            pricing.get("price_1m_blended_3_to_1"),
        )
