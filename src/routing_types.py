from dataclasses import dataclass
from enum import StrEnum


class TaskFamily(StrEnum):
    SIMPLE_QA = "simple_qa"
    SUMMARIZATION = "summarization"
    LONG_CONTEXT_READING = "long_context_reading"
    GENERAL_ANALYSIS = "general_analysis"
    CODING_BASIC = "coding_basic"
    CODING_ANALYSIS = "coding_analysis"
    CODING_DEBUG = "coding_debug"
    STRUCTURED_DATA = "structured_data"


class SelectionPolicy(StrEnum):
    BALANCED = "balanced"
    COST_SAVER = "cost_saver"
    QUALITY_FIRST = "quality_first"
    LATENCY_FIRST = "latency_first"
    NOPREF = "nopref"


@dataclass(frozen=True)
class ModelSelection:
    task_family: TaskFamily
    selection_policy: SelectionPolicy
    requires_reasoning: bool
    reasoning_effort: str
    difficulty_score: int
    complexity_score: int
    coding_score: int
    context_score: int
    has_code: bool
    has_stacktrace: bool
    has_structured_data: bool
    asks_for_analysis: bool
    token_count: int
    reason: str
