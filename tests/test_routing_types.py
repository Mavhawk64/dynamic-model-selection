from src.routing_types import ModelSelection, SelectionPolicy, TaskFamily


def test_selection_policy_values() -> None:
    assert SelectionPolicy.BALANCED == "balanced"
    assert SelectionPolicy.COST_SAVER == "cost_saver"
    assert SelectionPolicy.QUALITY_FIRST == "quality_first"
    assert SelectionPolicy.LATENCY_FIRST == "latency_first"
    assert SelectionPolicy.NOPREF == "nopref"


def test_task_family_values() -> None:
    assert TaskFamily.SIMPLE_QA == "simple_qa"
    assert TaskFamily.CODING_DEBUG == "coding_debug"


def test_model_selection_constructs() -> None:
    selection = ModelSelection(
        task_family=TaskFamily.SIMPLE_QA,
        selection_policy=SelectionPolicy.NOPREF,
        requires_reasoning=False,
        reasoning_effort="none",
        difficulty_score=0,
        complexity_score=0,
        coding_score=0,
        context_score=0,
        has_code=False,
        has_stacktrace=False,
        has_structured_data=False,
        asks_for_analysis=False,
        token_count=10,
        reason="simple prompt",
    )

    assert selection.task_family == TaskFamily.SIMPLE_QA
    assert selection.selection_policy == SelectionPolicy.NOPREF
    assert selection.token_count == 10
