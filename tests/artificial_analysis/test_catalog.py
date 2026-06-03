from datetime import date

import pytest

from src.artificial_analysis.catalog import (
    ModelCandidate,
    normalize_model,
    normalize_models,
    parse_date,
)


# --- parse_date ---


def test_parse_date_valid() -> None:
    assert parse_date("2026-01-01") == date(2026, 1, 1)


def test_parse_date_none() -> None:
    assert parse_date(None) is None


def test_parse_date_empty_string() -> None:
    assert parse_date("") is None


# --- normalize_model ---


def _raw_model(**overrides) -> dict:
    base: dict = {
        "id": "test-model",
        "name": "Test Model",
        "model_creator": {"name": "TestCo"},
        "evaluations": {
            "artificial_analysis_intelligence_index": 85.0,
            "artificial_analysis_coding_index": 90.0,
        },
        "pricing": {
            "price_1m_blended_3_to_1": 5.0,
            "price_1m_input_tokens": 3.0,
            "price_1m_output_tokens": 15.0,
        },
        "median_output_tokens_per_second": 100.0,
        "median_time_to_first_token_seconds": 0.5,
        "release_date": "2026-01-01",
    }
    base.update(overrides)
    return base


def test_normalize_model_full_fields() -> None:
    result = normalize_model(_raw_model())

    assert result.id == "test-model"
    assert result.name == "Test Model"
    assert result.provider == "TestCo"
    assert result.intelligence_index == 85.0
    assert result.coding_index == 90.0
    assert result.blended_cost_per_1m == 5.0
    assert result.input_cost_per_1m == 3.0
    assert result.output_cost_per_1m == 15.0
    assert result.output_tokens_per_second == 100.0
    assert result.time_to_first_token_seconds == 0.5
    assert result.release_date == date(2026, 1, 1)


def test_normalize_model_missing_optional_fields() -> None:
    result = normalize_model({"id": "min-model", "name": "Minimal"})

    assert result.provider == "Unknown"
    assert result.intelligence_index is None
    assert result.coding_index is None
    assert result.blended_cost_per_1m is None
    assert result.input_cost_per_1m is None
    assert result.output_cost_per_1m is None
    assert result.output_tokens_per_second is None
    assert result.time_to_first_token_seconds is None
    assert result.release_date is None


def test_normalize_model_null_evaluations() -> None:
    result = normalize_model(_raw_model(evaluations=None))

    assert result.intelligence_index is None
    assert result.coding_index is None


def test_normalize_model_null_pricing() -> None:
    result = normalize_model(_raw_model(pricing=None))

    assert result.blended_cost_per_1m is None
    assert result.input_cost_per_1m is None
    assert result.output_cost_per_1m is None


def test_normalize_model_null_creator() -> None:
    result = normalize_model(_raw_model(model_creator=None))

    assert result.provider == "Unknown"


def test_normalize_model_returns_frozen_dataclass() -> None:
    result = normalize_model(_raw_model())

    assert isinstance(result, ModelCandidate)
    with pytest.raises(Exception):
        result.name = "mutated"  # type: ignore[misc]


# --- normalize_models ---


def test_normalize_models_empty_data() -> None:
    assert normalize_models({"data": []}) == []


def test_normalize_models_missing_data_key() -> None:
    assert normalize_models({}) == []


def test_normalize_models_multiple_items() -> None:
    data = {
        "data": [
            _raw_model(id="alpha", name="Alpha"),
            _raw_model(id="beta", name="Beta"),
        ]
    }

    result = normalize_models(data)

    assert len(result) == 2
    assert result[0].name == "Alpha"
    assert result[1].name == "Beta"
