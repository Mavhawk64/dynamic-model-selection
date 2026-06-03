import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

ANALYSIS_PATTERNS = [
    r"\banaly[sz]e\b",
    r"\bcompare\b",
    r"\bevaluate\b",
    r"\bassess\b",
    r"\btrade[- ]?offs?\b",
    r"\bpros and cons\b",
    r"\badvantages?\b",
    r"\bdisadvantages?\b",
    r"\brecommend\b",
    r"\bbest approach\b",
    r"\bstrategy\b",
    r"\barchitecture\b",
    r"\bdesign\b",
    r"\bdiagnose\b",
    r"\bdebug\b",
    r"\broot cause\b",
    r"\bwhy\b",
    r"\bexplain\b",
    r"\bderive\b",
    r"\bprove\b",
    r"\boptimi[sz]e\b",
    r"\bfix\b",
]


QUESTION_RE = re.compile(r"\?")
ANALYSIS_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in ANALYSIS_PATTERNS
]


@dataclass(frozen=True)
class TextComplexitySignals:
    char_count: int
    word_count: int
    token_count: int
    sentence_count: int
    question_count: int
    asks_for_analysis: bool
    analysis_score: int
    complexity_score: int
    likely_needs_thinking: bool


@lru_cache(maxsize=16)
def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_openai_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_sentences_basic(text: str) -> int:
    sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text.strip())
    return len([sentence for sentence in sentences if sentence.strip()])


def count_questions(text: str) -> int:
    return len(QUESTION_RE.findall(text))


def analysis_score(text: str) -> int:
    return sum(1 for regex in ANALYSIS_REGEXES if regex.search(text))


def asks_for_analysis(text: str) -> bool:
    return analysis_score(text) > 0


def complexity_score(
    text: str,
    *,
    model: str = "gpt-4o",
    has_code: bool = False,
    has_stacktrace: bool = False,
    has_structured_data: bool = False,
) -> int:
    tokens = count_openai_tokens(text, model)
    questions = count_questions(text)
    analysis = analysis_score(text)

    score = 0

    if tokens >= 500:
        score += 1
    if tokens >= 1500:
        score += 2
    if tokens >= 4000:
        score += 3

    if questions >= 2:
        score += 1
    if questions >= 5:
        score += 2

    if analysis:
        score += 4

    if has_code:
        score += 3
    if has_stacktrace:
        score += 4
    if has_structured_data:
        score += 1

    return score


def likely_needs_thinking(
    text: str,
    *,
    model: str = "gpt-4o",
    has_code: bool = False,
    has_stacktrace: bool = False,
    has_structured_data: bool = False,
) -> bool:
    return (
        complexity_score(
            text,
            model=model,
            has_code=has_code,
            has_stacktrace=has_stacktrace,
            has_structured_data=has_structured_data,
        )
        >= 4
    )


def analyze_text_complexity(
    text: str,
    *,
    model: str = "gpt-4o",
    has_code: bool = False,
    has_stacktrace: bool = False,
    has_structured_data: bool = False,
) -> TextComplexitySignals:
    score = complexity_score(
        text,
        model=model,
        has_code=has_code,
        has_stacktrace=has_stacktrace,
        has_structured_data=has_structured_data,
    )

    return TextComplexitySignals(
        char_count=len(text),
        word_count=count_words(text),
        token_count=count_openai_tokens(text, model),
        sentence_count=count_sentences_basic(text),
        question_count=count_questions(text),
        asks_for_analysis=asks_for_analysis(text),
        analysis_score=analysis_score(text),
        complexity_score=score,
        likely_needs_thinking=score >= 4,
    )


if __name__ == "__main__":
    examples = [
        "What is the capital of France?",
        "Can you compare Rust and Go for backend services?",
        "Please analyze the tradeoffs of using a cache here.",
        "Debug this traceback and explain the root cause.",
        "Summarize this paragraph in one sentence.",
        ("I have a long request. " * 200 + "Can you recommend the best approach?"),
    ]

    for example in examples:
        print("=" * 80)
        print(example[:77] + "..." if len(example) >= 80 else example)
        print(analyze_text_complexity(example))
