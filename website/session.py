"""A notebook harness (robot_metrics' docs/session.py, copied as is for the landing page): run code cells in one namespace and keep each top-level statement with its own output.

    session = Session({})
    session.cell("label", "caption", "total = 1 + 1\\ntotal")
    session.cells          # [{"label", "caption", "code", "output", "seconds", "statements"}]
    session.cells[0]["statements"]   # [{"code", "output", "gap_before"}]: one per top-level statement
    session.transcript()   # the cells as a terminal would show them (>>> / ... then each statement's output)

The code shown is exactly the code that ran. A cell is split with ``ast`` into its top-level statements; each runs on
its own and keeps what it printed, and a bare expression echoes its repr as the REPL does. A compound statement (a
``for`` loop, a ``def``) stays whole, with the comments and blank lines inside it. ``gap_before`` records a blank line
in the source before a statement: the report draws it as the space between two ideas.

WHY one statement then its output (Zach, /rate bad 2026-11-18, on the metrics-API gallery): "make it more like line by
line so I can see the output of each line ... using white spacing within blocks is important to visually separate
concepts." Ported from the metrics-API proposals' ``_session.py`` (robot_models f0daa34), then split per statement.
"""

from __future__ import annotations

import ast
import contextlib
import io
import time
from typing import Any


class Session:
    def __init__(self, namespace: dict[str, Any]) -> None:
        self.namespace = namespace
        self.cells: list[dict[str, Any]] = []

    def cell(self, label: str, caption: str, code: str) -> None:
        code = code.strip("\n")
        lines = code.splitlines()
        tree = ast.parse(code)
        statements, previous_end = [], 0
        for node in tree.body:
            start = min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])])
            if start <= previous_end:
                # WHY refuse: one statement per line is the rule this harness exists for
                raise ValueError(f"cell {label!r}, line {start}: two statements share a line")
            between = lines[previous_end:start - 1]           # blank lines and comments before this statement
            leading = next((index for index, line in enumerate(between) if line.strip()), len(between))
            source = "\n".join(between[leading:] + lines[start - 1:node.end_lineno])
            statements.append({"node": node, "code": source, "gap_before": bool(statements) and leading > 0})
            previous_end = node.end_lineno
        compiled = [self._compile(statement["node"], label) for statement in statements]   # before any runs: timing
        started = time.perf_counter()
        for statement, (program, is_expression) in zip(statements, compiled):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                if is_expression:
                    value = eval(program, self.namespace)
                    if value is not None:
                        print(repr(value))
                else:
                    exec(program, self.namespace)
            statement["output"] = buffer.getvalue().rstrip()
            del statement["node"]
        output = "\n".join(statement["output"] for statement in statements if statement["output"])
        self.cells.append({"label": label, "caption": caption, "code": code, "output": output,
                           "seconds": f"{time.perf_counter() - started:.2f}", "statements": statements})

    @staticmethod
    def _compile(node: ast.stmt, label: str) -> tuple[Any, bool]:
        if isinstance(node, ast.Expr):
            return compile(ast.Expression(node.value), f"<cell {label}>", "eval"), True
        return compile(ast.Module(body=[node], type_ignores=[]), f"<cell {label}>", "exec"), False

    @staticmethod
    def prompt(code: str) -> str:
        return "\n".join(("... " if line.startswith((" ", ")", "]", "}")) or not line.strip() else ">>> ") + line
                         for line in code.splitlines())

    def transcript(self) -> str:
        parts = []
        for number, cell in enumerate(self.cells, start=1):
            lines = [f"# {number} · {cell['label']}   ({cell['seconds']} s)"]
            for statement in cell["statements"]:
                if statement["gap_before"]:
                    lines.append("")
                lines.append(self.prompt(statement["code"]))
                if statement["output"]:
                    lines.append(statement["output"])
            parts.append("\n".join(lines))
        return "\n\n".join(parts)
