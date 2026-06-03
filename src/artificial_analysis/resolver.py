import json
from dataclasses import dataclass
from typing import List

try:
    from src.artificial_analysis.catalog import ModelCandidate
    from src.text_model_selector import ModelSelection, SelectionPolicy
except ImportError:
    from artificial_analysis.catalog import ModelCandidate
    from text_model_selector import ModelSelection, SelectionPolicy


@dataclass(frozen=True)
class ResolvedModel:
    candidate: ModelCandidate
    score: float
    benchmark_used: str
    reason: str

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(
            {
                "candidate": self.candidate.__dict__,
                "score": self.score,
                "benchmark_used": self.benchmark_used,
                "reason": self.reason,
            },
            indent=indent,
            default=str,
        )

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.__dict__,
            "score": self.score,
            "benchmark_used": self.benchmark_used,
            "reason": self.reason,
        }

    def get_name(self) -> str:
        return self.candidate.name

    def get_company(self) -> str:
        return self.candidate.provider

    def get_name_and_co(self) -> str:
        return f"{self.get_name()} by {self.get_company()}"


@dataclass(frozen=True)
class ResolvedModels:
    balanced: ResolvedModel
    cost_saver: ResolvedModel
    quality_first: ResolvedModel
    latency_first: ResolvedModel

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(
            {
                "balanced": self.balanced.to_dict(),
                "cost_saver": self.cost_saver.to_dict(),
                "quality_first": self.quality_first.to_dict(),
                "latency_first": self.latency_first.to_dict(),
            },
            indent=indent,
            default=str,
        )

    def get_name(self) -> List[dict[str, str]]:
        return [
            {"balanced": self.balanced.get_name()},
            {"cost_saver": self.cost_saver.get_name()},
            {"quality_first": self.quality_first.get_name()},
            {"latency_first": self.latency_first.get_name()},
        ]

    def get_name_and_co(self) -> List[dict[str, str]]:
        return [
            {
                "balanced": self.balanced.get_name(),
                "company": self.balanced.get_company(),
            },
            {
                "cost_saver": self.cost_saver.get_name(),
                "company": self.cost_saver.get_company(),
            },
            {
                "quality_first": self.quality_first.get_name(),
                "company": self.quality_first.get_company(),
            },
            {
                "latency_first": self.latency_first.get_name(),
                "company": self.latency_first.get_company(),
            },
        ]


def required_quality(selection: ModelSelection) -> float:
    if selection.difficulty_score >= 80:
        return 0.95
    if selection.difficulty_score >= 60:
        return 0.90
    if selection.difficulty_score >= 40:
        return 0.85
    if selection.difficulty_score >= 20:
        return 0.75
    return 0.60


def benchmark_score(
    candidate: ModelCandidate,
    selection: ModelSelection,
) -> tuple[str, float | None]:
    if selection.has_code or selection.has_stacktrace:
        return "coding_index", candidate.coding_index

    return "intelligence_index", candidate.intelligence_index


def score_candidates(
    selection: ModelSelection,
    candidates: list[ModelCandidate],
) -> list[tuple[ModelCandidate, str, float]]:
    scored: list[tuple[ModelCandidate, str, float]] = []

    for candidate in candidates:
        benchmark_name, benchmark = benchmark_score(candidate, selection)

        if benchmark is None:
            continue
        if candidate.blended_cost_per_1m is None:
            continue
        if candidate.time_to_first_token_seconds is None:
            continue

        scored.append((candidate, benchmark_name, benchmark))

    return scored


def viable_candidates(
    selection: ModelSelection,
    candidates: list[ModelCandidate],
) -> list[tuple[ModelCandidate, str, float]]:
    scored = score_candidates(selection, candidates)

    if not scored:
        raise ValueError("No usable model candidates found.")

    best_benchmark = max(score for _, _, score in scored)
    min_quality = best_benchmark * required_quality(selection)

    viable = [
        (candidate, benchmark_name, benchmark)
        for candidate, benchmark_name, benchmark in scored
        if benchmark >= min_quality
    ]

    if not viable:
        raise ValueError("No viable model candidates found.")

    return viable


def filter_big3(
    candidates: list[tuple[ModelCandidate, str, float]],
) -> list[tuple[ModelCandidate, str, float]]:
    big3_providers = {"OpenAI", "Anthropic", "Google"}
    return [
        (candidate, benchmark_name, benchmark)
        for candidate, benchmark_name, benchmark in candidates
        if candidate.provider in big3_providers
    ]


def resolve_model(
    selection: ModelSelection,
    candidates: list[ModelCandidate],
    policy: SelectionPolicy,
    big3_only: bool = False,
) -> ResolvedModel:
    if policy == SelectionPolicy.NOPREF:
        raise ValueError("Use resolve_models() when policy is NOPREF.")

    viable = viable_candidates(selection, candidates)

    if big3_only:
        viable = filter_big3(viable)

    if policy == SelectionPolicy.QUALITY_FIRST:
        chosen = max(viable, key=lambda item: item[2])

    elif policy == SelectionPolicy.COST_SAVER:
        chosen = min(
            viable,
            key=lambda item: item[0].blended_cost_per_1m or float("inf"),
        )

    elif policy == SelectionPolicy.LATENCY_FIRST:
        chosen = min(
            viable,
            key=lambda item: item[0].time_to_first_token_seconds or float("inf"),
        )

    elif policy == SelectionPolicy.BALANCED:
        chosen = max(
            viable,
            key=lambda item: (
                (item[2] / max(item[0].blended_cost_per_1m or 999999, 0.01))
                - 0.05 * (item[0].time_to_first_token_seconds or 0)
            ),
        )

    else:
        raise ValueError(f"Unsupported selection policy: {policy}")

    candidate, benchmark_name, benchmark = chosen

    return ResolvedModel(
        candidate=candidate,
        score=benchmark,
        benchmark_used=benchmark_name,
        reason=(
            f"Selected by {policy}; "
            f"{benchmark_name}={benchmark}; "
            f"difficulty={selection.difficulty_score}; "
            f"cost=${candidate.blended_cost_per_1m}/1M blended tokens; "
            f"ttft={candidate.time_to_first_token_seconds}s"
        ),
    )


def resolve_models(
    selection: ModelSelection,
    candidates: list[ModelCandidate],
    big3_only: bool = False,
) -> ResolvedModels:
    return ResolvedModels(
        balanced=resolve_model(
            selection,
            candidates,
            SelectionPolicy.BALANCED,
            big3_only=big3_only,
        ),
        cost_saver=resolve_model(
            selection,
            candidates,
            SelectionPolicy.COST_SAVER,
            big3_only=big3_only,
        ),
        quality_first=resolve_model(
            selection,
            candidates,
            SelectionPolicy.QUALITY_FIRST,
            big3_only=big3_only,
        ),
        latency_first=resolve_model(
            selection,
            candidates,
            SelectionPolicy.LATENCY_FIRST,
            big3_only=big3_only,
        ),
    )


def resolve(
    selection: ModelSelection,
    candidates: list[ModelCandidate],
    big3_only: bool = False,
) -> ResolvedModel | ResolvedModels:
    if selection.selection_policy == SelectionPolicy.NOPREF:
        return resolve_models(selection, candidates, big3_only=big3_only)

    return resolve_model(
        selection,
        candidates,
        selection.selection_policy,
        big3_only=big3_only,
    )
