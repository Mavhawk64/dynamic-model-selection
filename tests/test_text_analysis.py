import pytest

from src.text_analysis import (
    analysis_score,
    analyze_text_complexity,
    asks_for_analysis,
    complexity_score,
    count_openai_tokens,
    count_questions,
    count_sentences_basic,
    count_words,
    likely_needs_thinking,
)


def test_count_words() -> None:
    assert count_words("Hello, world! This is a test.") == 6


def test_count_questions() -> None:
    assert count_questions("What is this? Why does it happen?") == 2


@pytest.mark.parametrize(
    ("text", "minimum"),
    [
        ("Hello world.", 1),
        ("Hello world. This is another sentence.", 2),
        ("What is this? Why?", 2),
    ],
)
def test_count_sentences_basic(text: str, minimum: int) -> None:
    assert count_sentences_basic(text) >= minimum


def test_count_openai_tokens_returns_positive_count() -> None:
    assert count_openai_tokens("Hello world") > 0


@pytest.mark.parametrize(
    "text",
    [
        "Can you analyze this?",
        "Compare Rust and Go.",
        "Evaluate the tradeoffs.",
        "What is the root cause?",
        "Can you recommend the best approach?",
        "Please explain why this happens.",
        "Derive the equation.",
        "Prove this statement.",
        "Optimize this workflow.",
    ],
)
def test_asks_for_analysis_true(text: str) -> None:
    assert asks_for_analysis(text) is True
    assert analysis_score(text) > 0


@pytest.mark.parametrize(
    "text",
    [
        "Translate this sentence.",
        "Summarize this paragraph.",
        "Rewrite this email.",
        "What is the capital of France?",
        "Hello there.",
    ],
)
def test_asks_for_analysis_false_or_weak(text: str) -> None:
    assert asks_for_analysis(text) is False


def test_analyze_text_complexity_simple_query() -> None:
    result = analyze_text_complexity("What is the capital of France?")

    assert result.char_count > 0
    assert result.word_count > 0
    assert result.token_count > 0
    assert result.question_count == 1
    assert result.asks_for_analysis is False
    assert result.likely_needs_thinking is False


def test_analyze_text_complexity_analysis_query() -> None:
    result = analyze_text_complexity(
        "Can you compare Rust and Go and recommend the best backend architecture?"
    )

    assert result.asks_for_analysis is True
    assert result.analysis_score > 0
    assert result.likely_needs_thinking is True


def test_likely_needs_thinking_with_code_signal() -> None:
    assert likely_needs_thinking("Can you fix this?", has_code=True) is True


def test_likely_needs_thinking_with_stacktrace_signal() -> None:
    assert likely_needs_thinking("This failed.", has_stacktrace=True) is True


def test_structured_data_increases_complexity_but_not_necessarily_thinking() -> None:
    result = analyze_text_complexity(
        '{"name": "Maverick"}',
        has_structured_data=True,
    )

    assert result.complexity_score >= 1


def test_long_prompt_increases_complexity() -> None:
    text = "This is a long prompt. " * 500

    result = analyze_text_complexity(text)

    assert result.token_count >= 500
    assert result.complexity_score > 0


# --- edge cases ---


def test_count_words_empty_string() -> None:
    assert count_words("") == 0


def test_count_questions_zero() -> None:
    assert count_questions("No question here.") == 0


def test_count_sentences_basic_empty() -> None:
    assert count_sentences_basic("") == 0


# --- complexity_score direct ---


def test_complexity_score_analysis_keyword() -> None:
    assert complexity_score("analyze this") >= 4


def test_complexity_score_has_code_bonus() -> None:
    assert complexity_score("hello", has_code=True) == 3


def test_complexity_score_has_stacktrace_bonus() -> None:
    assert complexity_score("hello", has_stacktrace=True) == 4


def test_complexity_score_has_structured_data_bonus() -> None:
    assert complexity_score("hello", has_structured_data=True) == 1


def test_complexity_score_multiple_questions_adds_bonus() -> None:
    assert complexity_score("? ?") >= 1
