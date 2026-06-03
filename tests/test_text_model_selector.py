import pytest

from src.routing_types import SelectionPolicy, TaskFamily
from src.text_model_selector import (
    clamp,
    context_score_from_tokens,
    reasoning_effort_from_score,
    select_text_model,
)

FENCE = "`" * 3


def fenced(lang: str, body: str) -> str:
    return f"{FENCE}{lang}\n{body}\n{FENCE}"


# --- clamp ---


def test_clamp_within_range() -> None:
    assert clamp(50) == 50


def test_clamp_below_minimum() -> None:
    assert clamp(-10) == 0


def test_clamp_above_maximum() -> None:
    assert clamp(200) == 100


def test_clamp_custom_bounds() -> None:
    assert clamp(5, minimum=10, maximum=20) == 10
    assert clamp(25, minimum=10, maximum=20) == 20


# --- context_score_from_tokens ---


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (0, 0),
        (499, 0),
        (500, 25),
        (1499, 25),
        (1500, 50),
        (3999, 50),
        (4000, 75),
        (7999, 75),
        (8000, 100),
    ],
)
def test_context_score_from_tokens(tokens: int, expected: int) -> None:
    assert context_score_from_tokens(tokens) == expected


# --- reasoning_effort_from_score ---


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "none"),
        (1, "low"),
        (39, "low"),
        (40, "medium"),
        (74, "medium"),
        (75, "high"),
        (100, "high"),
    ],
)
def test_reasoning_effort_from_score(score: int, expected: str) -> None:
    assert reasoning_effort_from_score(score) == expected


# --- select_text_model task family routing ---


def test_simple_prompt_routes_to_simple_qa() -> None:
    result = select_text_model("What is the capital of France?")

    assert result.task_family == TaskFamily.SIMPLE_QA
    assert result.requires_reasoning is False
    assert result.reasoning_effort == "none"
    assert result.has_code is False
    assert result.has_stacktrace is False


def test_analysis_prompt_routes_to_general_analysis() -> None:
    result = select_text_model(
        "Can you compare Rust and Go and recommend the best backend architecture?"
    )

    assert result.task_family == TaskFamily.GENERAL_ANALYSIS
    assert result.requires_reasoning is True
    assert result.asks_for_analysis is True


def test_code_without_analysis_routes_to_coding_basic() -> None:
    result = select_text_model(fenced("python", "def foo():\n    pass"))

    assert result.task_family == TaskFamily.CODING_BASIC
    assert result.requires_reasoning is False
    assert result.has_code is True


def test_code_with_analysis_routes_to_coding_analysis() -> None:
    result = select_text_model(
        "Can you analyze this code?\n\n"
        + fenced("rust", 'pub fn main() {\n    println!("hello");\n}')
    )

    assert result.task_family == TaskFamily.CODING_ANALYSIS
    assert result.requires_reasoning is True
    assert result.has_code is True
    assert result.asks_for_analysis is True


def test_stacktrace_routes_to_coding_debug() -> None:
    result = select_text_model(
        "Traceback (most recent call last):\n"
        '  File "main.py", line 1, in <module>\n'
        "ValueError: bad value"
    )

    assert result.task_family == TaskFamily.CODING_DEBUG
    assert result.requires_reasoning is True
    assert result.has_stacktrace is True


def test_long_prompt_routes_to_long_context_reading() -> None:
    result = select_text_model("This is background context. " * 600)

    assert result.task_family == TaskFamily.LONG_CONTEXT_READING
    assert result.token_count >= 1500


def test_structured_data_routes_to_structured_data() -> None:
    result = select_text_model('"name": "Maverick", "score": 99')

    assert result.task_family == TaskFamily.STRUCTURED_DATA
    assert result.has_structured_data is True


def test_inline_code_routes_to_coding_basic() -> None:
    result = select_text_model("Run `git status` and paste the output.")

    assert result.has_code is True
    assert result.task_family == TaskFamily.CODING_BASIC


# --- selection_policy propagation ---


def test_selection_policy_is_propagated() -> None:
    result = select_text_model(
        "What is the capital of France?",
        selection_policy=SelectionPolicy.COST_SAVER,
    )

    assert result.selection_policy == SelectionPolicy.COST_SAVER


def test_default_selection_policy_is_nopref() -> None:
    result = select_text_model("What is the capital of France?")

    assert result.selection_policy == SelectionPolicy.NOPREF
