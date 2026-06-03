import pytest

from src.code_analysis import (
    analyze_code_signals,
    fenced_blocks,
    has_fenced_code,
    has_inline_code,
    has_stacktrace,
    has_structured_data,
    inline_code_segments,
    looks_like_code,
    remove_fenced_blocks,
)

FENCE = "`" * 3


def fenced(lang: str, body: str) -> str:
    return f"{FENCE}{lang}\n{body}\n{FENCE}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (fenced("python", "def foo():\n    pass"), True),
        (
            fenced(
                "rust",
                "pub fn foo() -> Result<(), Error> {\n    Ok(())\n}",
            ),
            True,
        ),
        (
            fenced(
                "java",
                "public class Main {\n    public static void main(String[] args) {}\n}",
            ),
            True,
        ),
        (fenced("text", "This is plain text."), False),
        (fenced("markdown", "# Heading"), False),
        (fenced("mermaid", "flowchart TD\nA --> B"), False),
    ],
)
def test_has_fenced_code(text: str, expected: bool):
    assert has_fenced_code(text) is expected


def test_unlabeled_python_fence():
    text = fenced("", 'def bar():\n    print("Hello")')

    assert has_fenced_code(text) is True


def test_unlabeled_plain_text_fence():
    text = fenced("", "This is just a sentence.")

    assert has_fenced_code(text) is False


def test_fenced_blocks_extract():
    text = "Before\n\n" + fenced("python", 'print("hello")') + "\n\nAfter"

    assert fenced_blocks(text) == [("python", 'print("hello")')]


def test_remove_fenced_blocks():
    text = (
        "Before `inline`\n\n"
        + fenced("python", 'print("hello")')
        + "\n\nAfter `more_inline`"
    )

    cleaned = remove_fenced_blocks(text)

    assert "print" not in cleaned
    assert "`inline`" in cleaned
    assert "`more_inline`" in cleaned


def test_inline_code_segments():
    text = (
        "Use `git status`.\n\n"
        + fenced("python", "x = `not inline`")
        + "\n\nThen use `cargo test`."
    )

    assert inline_code_segments(text) == [
        "git status",
        "cargo test",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Run `git status`.", True),
        ("Run `cargo test`.", True),
        ("Use `print('hello')`.", True),
        ("The variable is `user_id`.", False),
        ("The class is `AgentProxyService`.", False),
        ("Check `src/main.rs`.", False),
    ],
)
def test_has_inline_code(text: str, expected: bool):
    assert has_inline_code(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "def foo():\n    pass",
        "class MyClass:",
        "import pathlib",
        "from pathlib import Path",
        "print('hello')",
        "if x > 0:",
        "for item in items:",
    ],
)
def test_python_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "public class Main {}",
        "public static void main(String[] args) {}",
        'System.out.println("hello");',
        "int count = 0;",
        "if (x > 0) { return; }",
    ],
)
def test_java_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "pub fn main() {}",
        "let mut count = 0;",
        "impl Display for Foo {}",
        "struct Person {}",
        "enum Color {}",
        "match value { Some(x) => x, None => 0 }",
        "Result<(), Error>",
    ],
)
def test_rust_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "func main() {}",
        "package main",
        "x := 10",
        "defer file.Close()",
    ],
)
def test_go_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "#include <stdio.h>",
        "#define MAX_SIZE 100",
        "std::vector<int> v;",
        "cout << value;",
        'printf("hello");',
        "template <typename T>",
    ],
)
def test_cpp_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "SELECT * FROM users",
        "INSERT INTO users VALUES (1)",
        "UPDATE users SET name='Mav'",
        "DELETE FROM users WHERE id=1",
        "CREATE TABLE users (id INT)",
    ],
)
def test_sql_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "git status",
        "cargo test",
        "docker build .",
        "kubectl get pods",
        "grep -R foo .",
        "cat file.txt >/dev/null",
    ],
)
def test_shell_detection(text: str):
    assert looks_like_code(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"name": "Maverick"}', True),
        ("{'name': 'Maverick'}", True),
        ('"name": "Maverick"', True),
        ("Name: Maverick", False),
        ("Question: why?", False),
        ("Note: this matters", False),
    ],
)
def test_structured_data_detection(text: str, expected: bool):
    assert has_structured_data(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        """Traceback (most recent call last):
  File "main.py", line 1
ValueError: bad value""",
        "TypeError: unsupported operand type(s)",
        "NullPointerException: something broke",
        "    at com.example.Main.main(Main.java:12)",
    ],
)
def test_stacktrace_detection(text: str):
    assert has_stacktrace(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hello world",
        "Can you explain inheritance?",
        "Name: Maverick",
        "Question: why does this happen?",
        "I think this is good; however, it might be wrong.",
        "The function (f) maps x to y.",
    ],
)
def test_plain_text_not_code(text: str):
    assert not looks_like_code(text)


def test_analyze_fenced_code():
    text = "Here is some code:\n\n" + fenced("python", "def foo():\n    pass")

    result = analyze_code_signals(text)

    assert result.has_code
    assert result.has_fenced_code
    assert result.fenced_block_count == 1


def test_analyze_inline_code():
    result = analyze_code_signals("Run `git status` and then `cargo test`.")

    assert result.has_code
    assert result.has_inline_code
    assert result.inline_code_count == 2


def test_analyze_plain_text():
    result = analyze_code_signals("This is just a normal sentence.")

    assert not result.has_code
    assert not result.has_fenced_code
    assert not result.has_inline_code
    assert not result.has_stacktrace


@pytest.mark.parametrize(
    "text",
    [
        "const x = 42;",
        "let name = 'hello';",
        "var count = 0;",
        "function greet(name) {}",
        'console.log("hello");',
        "if (x > 0) { return; }",
    ],
)
def test_javascript_detection(text: str):
    assert looks_like_code(text)


def test_analyze_code_signals_with_stacktrace():
    text = 'Traceback (most recent call last):\n  File "main.py", line 1\nValueError: bad value'

    result = analyze_code_signals(text)

    assert result.has_stacktrace
    assert result.has_code


def test_analyze_code_signals_with_structured_data():
    result = analyze_code_signals('{"name": "Maverick", "score": 99}')

    assert result.has_structured_data
