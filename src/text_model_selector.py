try:
    from src.code_analysis import analyze_code_signals
    from src.routing_types import ModelSelection, SelectionPolicy, TaskFamily
    from src.text_analysis import analyze_text_complexity
except ImportError:
    from code_analysis import analyze_code_signals
    from routing_types import ModelSelection, SelectionPolicy, TaskFamily
    from text_analysis import analyze_text_complexity


def clamp(value: int, minimum: int = 0, maximum: int = 100) -> int:
    return max(minimum, min(maximum, value))


def context_score_from_tokens(token_count: int) -> int:
    if token_count >= 8000:
        return 100
    if token_count >= 4000:
        return 75
    if token_count >= 1500:
        return 50
    if token_count >= 500:
        return 25
    return 0


def reasoning_effort_from_score(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 40:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def select_text_model(
    text: str,
    *,
    tokenizer_model: str = "gpt-4o",
    selection_policy: SelectionPolicy = SelectionPolicy.NOPREF,
) -> ModelSelection:
    code = analyze_code_signals(text)

    complexity = analyze_text_complexity(
        text,
        model=tokenizer_model,
        has_code=code.has_code,
        has_stacktrace=code.has_stacktrace,
        has_structured_data=code.has_structured_data,
    )

    token_count = complexity.token_count
    context_score = context_score_from_tokens(token_count)

    coding_score = 0
    if code.has_code:
        coding_score += 40
    if code.has_stacktrace:
        coding_score += 40
    if complexity.asks_for_analysis and code.has_code:
        coding_score += 20
    coding_score = clamp(coding_score)

    difficulty_score = clamp(
        complexity.complexity_score * 12 + context_score // 2 + coding_score // 2
    )

    requires_reasoning = (
        code.has_stacktrace
        or complexity.asks_for_analysis
        or complexity.likely_needs_thinking
    )

    reasoning_effort = reasoning_effort_from_score(
        difficulty_score if requires_reasoning else 0
    )

    if code.has_stacktrace:
        task_family = TaskFamily.CODING_DEBUG
        reason = "stacktrace/debugging signal detected"
    elif code.has_code and complexity.asks_for_analysis:
        task_family = TaskFamily.CODING_ANALYSIS
        reason = "code plus analysis intent detected"
    elif code.has_code:
        task_family = TaskFamily.CODING_BASIC
        reason = "code detected without strong reasoning signal"
    elif complexity.asks_for_analysis:
        task_family = TaskFamily.GENERAL_ANALYSIS
        reason = "analysis intent detected"
    elif code.has_structured_data:
        task_family = TaskFamily.STRUCTURED_DATA
        reason = "structured data detected"
    elif token_count >= 1500:
        task_family = TaskFamily.LONG_CONTEXT_READING
        reason = "long prompt without strong reasoning signal"
    else:
        task_family = TaskFamily.SIMPLE_QA
        reason = "simple prompt"

    return ModelSelection(
        task_family=task_family,
        selection_policy=selection_policy,
        requires_reasoning=requires_reasoning,
        reasoning_effort=reasoning_effort,
        difficulty_score=difficulty_score,
        complexity_score=complexity.complexity_score,
        coding_score=coding_score,
        context_score=context_score,
        has_code=code.has_code,
        has_stacktrace=code.has_stacktrace,
        has_structured_data=code.has_structured_data,
        asks_for_analysis=complexity.asks_for_analysis,
        token_count=token_count,
        reason=reason,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    def analyze_prompt(prompt: str, source: str) -> None:
        result = select_text_model(prompt)

        print("=" * 120)
        print(source)
        print("-" * 120)
        print(prompt[:300].replace("\n", " "))
        if len(prompt) > 300:
            print("...")
        print()
        print(result)
        print()

    if len(sys.argv) == 1:
        examples = [
            "What is the capital of France?",
            "Summarize this paragraph.",
            "Can you compare Rust and Go for backend APIs?",
            "Analyze the tradeoffs between PostgreSQL and MongoDB.",
            "Why is my application crashing?",
            """
            Traceback (most recent call last):
              File "main.py", line 1
            ValueError: bad value
            """,
            """
            ```python
            def foo():
                pass
            ```
            """,
        ]

        for i, example in enumerate(examples, start=1):
            analyze_prompt(example.strip(), f"builtin:{i}")

        sys.exit(0)

    path = Path(sys.argv[1])

    if path.is_file():
        analyze_prompt(path.read_text(), str(path))
        sys.exit(0)

    if path.is_dir():
        extensions = {
            ".txt",
            ".md",
            ".prompt",
            ".json",
        }

        for file in sorted(path.rglob("*")):
            if file.suffix.lower() not in extensions:
                continue

            try:
                analyze_prompt(
                    file.read_text(),
                    str(file),
                )
            except Exception as exc:
                print(f"ERROR: {file}: {exc}")

        sys.exit(0)

    raise FileNotFoundError(path)
