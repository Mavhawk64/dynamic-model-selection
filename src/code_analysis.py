import re
from dataclasses import dataclass

FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+#.-]*)[ \t]*\n(?P<body>.*?)```",
    re.DOTALL,
)

INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

KNOWN_CODE_LANGS = {
    "python",
    "py",
    "javascript",
    "js",
    "typescript",
    "ts",
    "java",
    "c",
    "h",
    "cpp",
    "c++",
    "cc",
    "hpp",
    "csharp",
    "cs",
    "go",
    "golang",
    "rust",
    "rs",
    "sql",
    "bash",
    "sh",
    "zsh",
    "shell",
    "powershell",
    "ps1",
    "json",
    "yaml",
    "yml",
    "toml",
    "html",
    "css",
    "xml",
    "tex",
    "latex",
    "dockerfile",
    "makefile",
}

NON_CODE_LANGS = {
    "text",
    "txt",
    "plain",
    "plaintext",
    "md",
    "markdown",
    "mmd",
    "mermaid",
    "csv",
    "tsv",
}

STRONG_CODE_PATTERNS = [
    # Python
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bfrom\s+\w+(\.\w+)*\s+import\b",
    r"\bimport\s+\w+",
    r"\bprint\s*\(",
    r"\bif\s+.+:",
    r"\bfor\s+.+:",
    r"\bwhile\s+.+:",
    r"\btry\s*:",
    r"\bexcept\b",
    # Java / C# / C / C++ / JS-like
    r"\b(public|private|protected)\s+(static\s+)?(class|interface|void|int|double|float|boolean|bool|char|String)\b",
    r"\b(int|double|float|boolean|bool|char|String|long|short|byte)\s+\w+\s*(=|;|,|\))",
    r"\bpublic\s+static\s+void\s+main\s*\(",
    r"\bSystem\.out\.println\s*\(",
    r"\bconsole\.log\s*\(",
    r"\bfunction\s+\w+\s*\(",
    r"\b(const|let|var)\s+\w+\s*=",
    r"\b(if|for|while|switch)\s*\(",
    r"\belse\s*\{",
    r"\bbreak\s*;",
    r"\bcontinue\s*;",
    r"#include\s*[<\"][\w./]+[>\"]",
    r"#define\s+\w+",
    r"\bstd::\w+",
    r"\bcout\s*<<",
    r"\bcin\s*>>",
    r"\bprintf\s*\(",
    r"\bscanf\s*\(",
    r"\bmalloc\s*\(",
    r"\bfree\s*\(",
    r"\btemplate\s*<",
    r"\btypename\s+\w+",
    r"\busing\s+namespace\s+std\b",
    # Rust
    r"\bfn\s+\w+\s*\(",
    r"\bpub\s+(fn|struct|enum|trait|mod|use|const|static)\b",
    r"\blet\s+mut\s+\w+",
    r"\blet\s+\w+\s*=",
    r"\bimpl\s+\w+",
    r"\btrait\s+\w+",
    r"\bstruct\s+\w+",
    r"\benum\s+\w+",
    r"\bmatch\s+.+\s*\{",
    r"\b(Some|Ok|Err)\s*\(",
    r"\bNone\b",
    r"\bResult\s*<",
    r"\bOption\s*<",
    r"->\s*[\w:<>,\s&']+",
    r"=>",
    # Go
    r"\bfunc\s+\w+\s*\(",
    r"\bpackage\s+\w+",
    r"\bimport\s*\(",
    r"\bdefer\s+\w+",
    r"\bgo\s+\w+\s*\(",
    r"\bchan\s+\w+",
    r":=",
    # SQL
    r"\bSELECT\s+.+\s+FROM\b",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+\w+\s+SET\b",
    r"\bDELETE\s+FROM\b",
    r"\bCREATE\s+TABLE\b",
    r"\bALTER\s+TABLE\b",
    # Shell / CLI
    r"^\s*(git|cd|ls|grep|find|cat|echo|python|pip|npm|cargo|go|java|javac|make|docker|kubectl)\s+",
    r"\b(export|alias)\s+\w+=",
    r"\$\(.+\)",
    r">/dev/null",
]

WEAK_CODE_PATTERNS = [
    r"\breturn\b",
    r"\bpass\b",
    r"\bmut\s+\w+",
    r"\buse\s+[\w:]+",
    r"\w+\.\w+\s*\(",
    r"\w+::\w+",
    r"\w+\s*=\s*[^=].+",
    r"[{}();]",
    r"\|\s*\w+",
    r"\$\w+",
]

STRUCTURED_DATA_PATTERNS = [
    r"^\s*[\{\[]",
    r'["\'][^"\']+["\']\s*:\s*',
]

STACKTRACE_PATTERNS = [
    r"\bTraceback \(most recent call last\):",
    r"\b[A-Za-z_][\w.]*Error:",
    r"\b[A-Za-z_][\w.]*Exception:",
    r"^\s*File \"[^\"]+\", line \d+",
    r"^\s*at\s+[\w.$]+\(",
]


STRONG_CODE_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in STRONG_CODE_PATTERNS
]

WEAK_CODE_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in WEAK_CODE_PATTERNS
]

STRUCTURED_DATA_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in STRUCTURED_DATA_PATTERNS
]

STACKTRACE_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in STACKTRACE_PATTERNS
]


@dataclass(frozen=True)
class TextCodeSignals:
    has_code: bool
    has_fenced_code: bool
    has_inline_code: bool
    has_stacktrace: bool
    has_structured_data: bool
    fenced_block_count: int
    inline_code_count: int
    code_score: int


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    return [
        (match.group("lang").strip().lower(), match.group("body").strip())
        for match in FENCE_RE.finditer(text)
    ]


def remove_fenced_blocks(text: str) -> str:
    return FENCE_RE.sub("", text)


def inline_code_segments(text: str) -> list[str]:
    text_without_fences = remove_fenced_blocks(text)
    return [segment.strip() for segment in INLINE_CODE_RE.findall(text_without_fences)]


def has_stacktrace(text: str) -> bool:
    return any(regex.search(text) for regex in STACKTRACE_REGEXES)


def has_structured_data(text: str) -> bool:
    return any(regex.search(text) for regex in STRUCTURED_DATA_REGEXES)


def code_score(text: str) -> int:
    stripped = text.strip()

    if not stripped:
        return 0

    score = 0

    score += 4 * sum(1 for regex in STRONG_CODE_REGEXES if regex.search(stripped))
    score += 1 * sum(1 for regex in WEAK_CODE_REGEXES if regex.search(stripped))
    score += 2 * sum(1 for regex in STRUCTURED_DATA_REGEXES if regex.search(stripped))
    score += 5 if has_stacktrace(stripped) else 0

    return score


def looks_like_code(text: str, threshold: int = 4) -> bool:
    return code_score(text) >= threshold


def has_fenced_code(text: str) -> bool:
    for lang, body in fenced_blocks(text):
        if not body:
            continue

        if lang in KNOWN_CODE_LANGS:
            return True

        if lang in NON_CODE_LANGS:
            continue

        if looks_like_code(body):
            return True

    return False


def has_inline_code(text: str) -> bool:
    for segment in inline_code_segments(text):
        if looks_like_code(segment, threshold=3):
            return True

    return False


def analyze_code_signals(text: str) -> TextCodeSignals:
    fenced = fenced_blocks(text)
    inline = inline_code_segments(text)

    fenced_code = has_fenced_code(text)
    inline_code = has_inline_code(text)
    stacktrace = has_stacktrace(text)
    structured_data = has_structured_data(text)
    score = code_score(remove_fenced_blocks(text))

    has_code = fenced_code or inline_code or stacktrace or score >= 4

    return TextCodeSignals(
        has_code=has_code,
        has_fenced_code=fenced_code,
        has_inline_code=inline_code,
        has_stacktrace=stacktrace,
        has_structured_data=structured_data,
        fenced_block_count=len(fenced),
        inline_code_count=len(inline),
        code_score=score,
    )


if __name__ == "__main__":
    examples = [
        "Hello world print('Hello world')",
        "Here is `print('hello')` inline.",
        """```python
def foo():
    pass
```""",
        """```text
This is just some text.
```""",
        """```
def bar():
    print("Hello, world!")
```""",
    ]

    for example in examples:
        print(analyze_code_signals(example))
