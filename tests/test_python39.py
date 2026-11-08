"""evalgen targets Python 3.10+ and also runs on 3.9, with byte-identical output (docs/DESIGN.md,
Reproducibility). This scan of evalgen, cli.py and the tests finds what Python 3.9 cannot run:

  - a type union written "X | Y" that is evaluated at runtime: a type alias, an isinstance argument,
    a default value. Annotations are exempt in modules that postpone them with
    ``from __future__ import annotations``, as every module here does;
  - a match statement, zip(strict=...), itertools.pairwise and dataclass(slots=..., kw_only=...).
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([*(ROOT / "evalgen").rglob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "cli.py"])
BUILTIN_TYPES = {"bool", "bytes", "dict", "float", "frozenset", "int", "list", "object", "set", "str", "tuple", "type"}


def _type_like(node: ast.AST) -> bool:
    """Whether an operand of "|" names a type: None, a builtin type, a CamelCase class (CONSTANTS and
    lower-case names are values, such as sets or regex flags), a subscripted type, or another union."""
    if isinstance(node, ast.Constant):
        return node.value is None
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
    if name is not None:
        return name in BUILTIN_TYPES or (name[0].isupper() and not name.isupper())
    if isinstance(node, ast.Subscript):
        return _type_like(node.value)
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and _type_like(node.left)


def _postponed_annotations(tree: ast.Module) -> set[int]:
    """The nodes inside annotations, when the module postpones their evaluation."""
    postponed = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                    and any(alias.name == "annotations" for alias in n.names) for n in tree.body)
    if not postponed:
        return set()
    roots: list[ast.AST | None] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            roots += [arg.annotation for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg) if arg]
            roots.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            roots.append(node.annotation)
    return {id(sub) for root in roots if root is not None for sub in ast.walk(root)}


def problems(path: Path) -> list[str]:
    """What Python 3.9 cannot run in one file, in line order."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, str(path))
    exempt = _postponed_annotations(tree)
    label = path.relative_to(ROOT) if ROOT in path.parents else path
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and id(node) not in exempt
                and _type_like(node.left) and _type_like(node.right)):
            found.append((node.lineno, f"a runtime type union, {ast.get_source_segment(source, node)}"))
        elif isinstance(node, ast.Match):
            found.append((node.lineno, "a match statement"))
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            keywords = {k.arg for k in node.keywords}
            if (name == "zip" and "strict" in keywords) or name == "pairwise" or (
                    name == "dataclass" and keywords & {"slots", "kw_only"}):
                found.append((node.lineno, ast.get_source_segment(source, node)))
    return [f"{label}:{line}: {what}" for line, what in sorted(found)]


def test_the_scan_finds_what_python39_cannot_run(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("from __future__ import annotations\nfrom typing import Callable, Optional\n"
                      "def f(x: int | None) -> dict[str, int] | None: ...\n"  # postponed: fine
                      "Builder = Callable[[str], int | None]\n"  # evaluated at import: Python 3.9 raises
                      "ok = Callable[[str], Optional[int]]\nflags = FLAG_A | FLAG_B\nboth = seen | {1}\n"
                      "pairs = zip(a, b, strict=True)\n")
    found = problems(sample)
    assert len(found) == 2 and "4: a runtime type union, int | None" in found[0] and "strict=True" in found[1]


def test_evalgen_and_its_tests_run_on_python39():
    assert SOURCES and [p for path in SOURCES for p in problems(path)] == []
