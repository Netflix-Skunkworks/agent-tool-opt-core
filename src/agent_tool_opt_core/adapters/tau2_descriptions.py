"""Extract and splice TauBench @is_tool docstrings without editing code.

The optimizer edits a small ``descriptions.md`` artifact. The full tools.py is
read-only context; splicing changes only existing docstring literal spans.
"""

from __future__ import annotations

import ast
import inspect


def _tool_functions(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source)
    methods = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                getattr(target, "id", None) == "is_tool"
                or getattr(target, "attr", None) == "is_tool"
            ):
                methods.append(node)
                break
    return methods


def tool_docstrings(source: str) -> dict[str, str]:
    docs = {}
    for method in _tool_functions(source):
        value = ast.get_docstring(method, clean=False)
        if value is not None:
            docs[method.name] = value
    return docs


def render_descriptions(source: str) -> str:
    parts = [
        "# Tool descriptions",
        "Improve the description under each heading so the agent calls the tool "
        "correctly. Keep every `## <tool>` heading exactly once. Edit only the "
        "text beneath it, keeping four spaces before every nonempty body line "
        "(including any embedded headings). Documentation may be rewritten or "
        "removed; keep the heading even when its body is empty.\n",
    ]
    for name, doc in tool_docstrings(source).items():
        body = "\n".join("    " + line for line in inspect.cleandoc(doc).split("\n"))
        parts.append(f"## {name}\n{body}\n")
    return "\n".join(parts)


def parse_descriptions(markdown: str, names: set[str]) -> dict[str, str]:
    """Reject missing/duplicate headings and ambiguous unindented body text."""
    descriptions: dict[str, str] = {}
    current: str | None = None
    body: list[str] = []

    def finish() -> None:
        if current is not None:
            descriptions[current] = "\n".join(body).strip("\n")

    for line in markdown.split("\n"):
        if line.startswith("## "):
            name = line[3:].strip()
            if name not in names:
                raise ValueError(f"unknown tool heading: {name}")
            if current is not None:
                finish()
            if name in descriptions:
                raise ValueError(f"duplicate tool heading: {name}")
            current, body = name, []
        elif current is not None:
            if line.startswith("    "):
                body.append(line[4:])
            elif not line.strip():
                body.append("")
            else:
                raise ValueError(f"description body for {current} must be indented")
    finish()
    missing = names - descriptions.keys()
    if missing:
        raise ValueError(f"missing tool headings: {', '.join(sorted(missing))}")
    return descriptions


def splice_docstrings(source: str, descriptions: dict[str, str]) -> str:
    """Replace only the existing @is_tool docstring literal byte spans."""
    encoded = source.encode("utf-8")
    offsets = [0]
    for line in encoded.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def offset(line: int, column: int) -> int:
        return offsets[line - 1] + column

    edits: list[tuple[int, int, bytes]] = []
    for method in _tool_functions(source):
        if method.name not in descriptions or not method.body:
            continue
        first = method.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        original = first.value.value
        updated = descriptions[method.name]
        if updated in (original, inspect.cleandoc(original)):
            continue
        value = first.value
        edits.append(
            (
                offset(value.lineno, value.col_offset),
                offset(value.end_lineno, value.end_col_offset),
                repr(updated).encode("utf-8"),
            )
        )
    for start, end, literal in sorted(edits, reverse=True):
        encoded = encoded[:start] + literal + encoded[end:]
    return encoded.decode("utf-8")
