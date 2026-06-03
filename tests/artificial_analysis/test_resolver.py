import json
from datetime import date

import pytest

from src.artificial_analysis.catalog import ModelCandidate
from src.artificial_analysis.resolver import (
    ResolvedModel,
    ResolvedModels,
    benchmark_score,
    filter_big3,
    required_quality,
    resolve,
    resolve_model,
    resolve_models,
    score_candidates,
    viable_candidates,
)
from src.routing_types import ModelSelection, SelectionPolicy, TaskFamily


def candidate(
    *,
    name: str,
    intelligence_index: float | None,
    coding_index: float | None,
    cost: float,
    latency: float,
) -> ModelCandidate:
    return ModelCandidate(
        id=name.lower().replace(" ", "-"),
        name=name,
        provider="TestProvider",
        intelligence_index=intelligence_index,
        coding_index=coding_index,
        blended_cost_per_1m=cost,
        input_cost_per_1m=cost,
        output_cost_per_1m=cost,
        output_tokens_per_second=100.0,
        time_to_first_token_seconds=latency,
        release_date=date(2026, 1, 1),
    )


def selection(
    *,
    has_code: bool = False,
    has_stacktrace: bool = False,
    difficulty_score: int = 30,
    selection_policy: SelectionPolicy = SelectionPolicy.NOPREF,
) -> ModelSelection:
    return ModelSelection(
        task_family=TaskFamily.CODING_DEBUG if has_stacktrace else TaskFamily.SIMPLE_QA,
        selection_policy=selection_policy,
        requires_reasoning=has_stacktrace,
        reasoning_effort="medium" if has_stacktrace else "none",
        difficulty_score=difficulty_score,
        complexity_score=3,
        coding_score=80 if has_stacktrace else 0,
        context_score=0,
        has_code=has_code,
        has_stacktrace=has_stacktrace,
        has_structured_data=False,
        asks_for_analysis=False,
        token_count=100,
        reason="test",
    )


def test_required_quality() -> None:
    assert required_quality(selection(difficulty_score=10)) == 0.60
    assert required_quality(selection(difficulty_score=20)) == 0.75
    assert required_quality(selection(difficulty_score=40)) == 0.85
    assert required_quality(selection(difficulty_score=60)) == 0.90
    assert required_quality(selection(difficulty_score=80)) == 0.95


def test_benchmark_score_uses_intelligence_for_non_code() -> None:
    model = candidate(
        name="Smart Model",
        intelligence_index=90.0,
        coding_index=50.0,
        cost=10.0,
        latency=5.0,
    )

    benchmark_name, score = benchmark_score(model, selection())

    assert benchmark_name == "intelligence_index"
    assert score == 90.0


def test_benchmark_score_uses_coding_for_code() -> None:
    model = candidate(
        name="Coding Model",
        intelligence_index=70.0,
        coding_index=95.0,
        cost=10.0,
        latency=5.0,
    )

    benchmark_name, score = benchmark_score(model, selection(has_code=True))

    assert benchmark_name == "coding_index"
    assert score == 95.0


def test_viable_candidates_filters_below_required_quality() -> None:
    models = [
        candidate(
            name="Bad",
            intelligence_index=10.0,
            coding_index=10.0,
            cost=1.0,
            latency=1.0,
        ),
        candidate(
            name="Good",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=10.0,
            latency=5.0,
        ),
    ]

    viable = viable_candidates(selection(difficulty_score=80), models)

    assert len(viable) == 1
    assert viable[0][0].name == "Good"


def test_cost_saver_selects_cheapest_viable() -> None:
    models = [
        candidate(
            name="Expensive",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=20.0,
            latency=5.0,
        ),
        candidate(
            name="Cheap",
            intelligence_index=85.0,
            coding_index=85.0,
            cost=1.0,
            latency=10.0,
        ),
    ]

    result = resolve_model(
        selection(difficulty_score=40), models, SelectionPolicy.COST_SAVER
    )

    assert result.candidate.name == "Cheap"


def test_quality_first_selects_highest_score() -> None:
    models = [
        candidate(
            name="Okay",
            intelligence_index=85.0,
            coding_index=85.0,
            cost=1.0,
            latency=1.0,
        ),
        candidate(
            name="Best",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=100.0,
            latency=20.0,
        ),
    ]

    result = resolve_model(
        selection(difficulty_score=40), models, SelectionPolicy.QUALITY_FIRST
    )

    assert result.candidate.name == "Best"


def test_latency_first_selects_fastest_viable() -> None:
    models = [
        candidate(
            name="Slow",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=1.0,
            latency=20.0,
        ),
        candidate(
            name="Fast",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=5.0,
            latency=1.0,
        ),
    ]

    result = resolve_model(
        selection(difficulty_score=40), models, SelectionPolicy.LATENCY_FIRST
    )

    assert result.candidate.name == "Fast"


def test_balanced_prefers_quality_per_dollar_with_latency_penalty() -> None:
    models = [
        candidate(
            name="Very Expensive",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=100.0,
            latency=1.0,
        ),
        candidate(
            name="Efficient",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=2.0,
            latency=2.0,
        ),
    ]

    result = resolve_model(
        selection(difficulty_score=40), models, SelectionPolicy.BALANCED
    )

    assert result.candidate.name == "Efficient"


def test_nopref_policy_rejected_by_resolve_model() -> None:
    models = [
        candidate(
            name="Model",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=1.0,
            latency=1.0,
        ),
    ]

    with pytest.raises(ValueError, match="Use resolve_models"):
        resolve_model(selection(), models, SelectionPolicy.NOPREF)


def test_resolve_models_returns_all_policies() -> None:
    models = [
        candidate(
            name="Cheap",
            intelligence_index=85.0,
            coding_index=85.0,
            cost=1.0,
            latency=10.0,
        ),
        candidate(
            name="Fast",
            intelligence_index=86.0,
            coding_index=86.0,
            cost=5.0,
            latency=1.0,
        ),
        candidate(
            name="Best",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=100.0,
            latency=20.0,
        ),
    ]

    result = resolve_models(selection(difficulty_score=40), models)

    assert isinstance(result, ResolvedModels)
    assert isinstance(result.balanced, ResolvedModel)
    assert isinstance(result.cost_saver, ResolvedModel)
    assert isinstance(result.quality_first, ResolvedModel)
    assert isinstance(result.latency_first, ResolvedModel)


def test_resolve_with_nopref_returns_resolved_models() -> None:
    models = [
        candidate(
            name="Cheap",
            intelligence_index=85.0,
            coding_index=85.0,
            cost=1.0,
            latency=10.0,
        ),
        candidate(
            name="Fast",
            intelligence_index=86.0,
            coding_index=86.0,
            cost=5.0,
            latency=1.0,
        ),
        candidate(
            name="Best",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=100.0,
            latency=20.0,
        ),
    ]

    result = resolve(selection(selection_policy=SelectionPolicy.NOPREF), models)

    assert isinstance(result, ResolvedModels)


def test_resolve_with_specific_policy_returns_single_model() -> None:
    models = [
        candidate(
            name="Cheap",
            intelligence_index=85.0,
            coding_index=85.0,
            cost=1.0,
            latency=10.0,
        ),
        candidate(
            name="Best",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=100.0,
            latency=20.0,
        ),
    ]

    result = resolve(
        selection(selection_policy=SelectionPolicy.QUALITY_FIRST),
        models,
    )

    assert isinstance(result, ResolvedModel)
    assert result.candidate.name == "Best"


def test_no_usable_candidates_raises() -> None:
    models = [
        candidate(
            name="Incomplete",
            intelligence_index=None,
            coding_index=None,
            cost=1.0,
            latency=1.0,
        ),
    ]

    with pytest.raises(ValueError, match="No usable model candidates"):
        viable_candidates(selection(), models)


# --- score_candidates filtering ---


def test_score_candidates_excludes_none_cost() -> None:
    models = [
        candidate(
            name="No Cost",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=0.0,
            latency=1.0,
        ),
    ]
    models[0] = ModelCandidate(
        id="no-cost",
        name="No Cost",
        provider="TestProvider",
        intelligence_index=90.0,
        coding_index=90.0,
        blended_cost_per_1m=None,
        input_cost_per_1m=None,
        output_cost_per_1m=None,
        output_tokens_per_second=100.0,
        time_to_first_token_seconds=1.0,
        release_date=date(2026, 1, 1),
    )

    assert score_candidates(selection(), models) == []


def test_score_candidates_excludes_none_latency() -> None:
    no_latency = ModelCandidate(
        id="no-latency",
        name="No Latency",
        provider="TestProvider",
        intelligence_index=90.0,
        coding_index=90.0,
        blended_cost_per_1m=5.0,
        input_cost_per_1m=5.0,
        output_cost_per_1m=5.0,
        output_tokens_per_second=100.0,
        time_to_first_token_seconds=None,
        release_date=date(2026, 1, 1),
    )

    assert score_candidates(selection(), [no_latency]) == []


# --- filter_big3 ---


def _provider_candidate(
    name: str,
    provider: str,
    intelligence: float = 90.0,
) -> ModelCandidate:
    return ModelCandidate(
        id=name.lower().replace(" ", "-"),
        name=name,
        provider=provider,
        intelligence_index=intelligence,
        coding_index=intelligence,
        blended_cost_per_1m=5.0,
        input_cost_per_1m=5.0,
        output_cost_per_1m=5.0,
        output_tokens_per_second=100.0,
        time_to_first_token_seconds=1.0,
        release_date=date(2026, 1, 1),
    )


def test_filter_big3_keeps_only_big3_providers() -> None:
    scored = [
        (_provider_candidate("GPT-4o", "OpenAI"), "intelligence_index", 90.0),
        (_provider_candidate("Claude", "Anthropic"), "intelligence_index", 88.0),
        (_provider_candidate("Gemini", "Google"), "intelligence_index", 87.0),
        (_provider_candidate("Llama", "Meta"), "intelligence_index", 95.0),
    ]

    result = filter_big3(scored)

    providers = {c.provider for c, _, _ in result}
    assert providers == {"OpenAI", "Anthropic", "Google"}
    assert len(result) == 3


def test_filter_big3_returns_empty_when_no_big3() -> None:
    scored = [
        (_provider_candidate("Llama", "Meta"), "intelligence_index", 95.0),
        (_provider_candidate("Mistral", "Mistral AI"), "intelligence_index", 80.0),
    ]

    assert filter_big3(scored) == []


def test_resolve_model_big3_only_excludes_non_big3() -> None:
    big3 = _provider_candidate("GPT-4o", "OpenAI", intelligence=88.0)
    non_big3 = _provider_candidate("Llama 4", "Meta", intelligence=95.0)

    result = resolve_model(
        selection(difficulty_score=40),
        [big3, non_big3],
        SelectionPolicy.QUALITY_FIRST,
        big3_only=True,
    )

    assert result.candidate.provider == "OpenAI"


# --- ResolvedModel methods ---


def _make_resolved_model() -> ResolvedModel:
    models = [
        candidate(
            name="Alpha",
            intelligence_index=90.0,
            coding_index=90.0,
            cost=5.0,
            latency=2.0,
        ),
    ]
    return resolve_model(
        selection(difficulty_score=20), models, SelectionPolicy.QUALITY_FIRST
    )


def test_resolved_model_get_name() -> None:
    result = _make_resolved_model()

    assert result.get_name() == "Alpha"


def test_resolved_model_get_company() -> None:
    result = _make_resolved_model()

    assert result.get_company() == "TestProvider"


def test_resolved_model_get_name_and_co() -> None:
    result = _make_resolved_model()

    assert result.get_name_and_co() == "Alpha by TestProvider"


def test_resolved_model_to_dict() -> None:
    result = _make_resolved_model()
    d = result.to_dict()

    assert "candidate" in d
    assert "score" in d
    assert "benchmark_used" in d
    assert "reason" in d
    assert d["score"] == 90.0


def test_resolved_model_to_json() -> None:
    result = _make_resolved_model()
    raw = result.to_json()

    parsed = json.loads(raw)
    assert parsed["score"] == 90.0
    assert "candidate" in parsed


# --- ResolvedModels methods ---


def _make_resolved_models() -> ResolvedModels:
    models = [
        candidate(
            name="Cheap",
            intelligence_index=85.0,
            coding_index=85.0,
            cost=1.0,
            latency=10.0,
        ),
        candidate(
            name="Fast",
            intelligence_index=86.0,
            coding_index=86.0,
            cost=5.0,
            latency=1.0,
        ),
        candidate(
            name="Best",
            intelligence_index=95.0,
            coding_index=95.0,
            cost=100.0,
            latency=20.0,
        ),
    ]
    return resolve_models(selection(difficulty_score=40), models)


def test_resolved_models_to_json() -> None:
    result = _make_resolved_models()
    parsed = json.loads(result.to_json())

    assert set(parsed.keys()) == {"balanced", "cost_saver", "quality_first", "latency_first"}


def test_resolved_models_get_name() -> None:
    result = _make_resolved_models()
    names = result.get_name()

    assert len(names) == 4
    assert all(isinstance(item, dict) for item in names)


def test_resolved_models_get_name_and_co() -> None:
    result = _make_resolved_models()
    entries = result.get_name_and_co()

    assert len(entries) == 4
    assert all("company" in item for item in entries)
