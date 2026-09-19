"""Every turn-gate permit must be taken through the capacity context managers.

``turn_slot`` and ``queued_turn_slot`` pair ``acquire`` with ``release`` in a
``finally``. A hand-rolled ``gate.try_acquire()`` anywhere else is one missing
``finally`` away from a permanently leaked slot, and at the SMALL profile's
limit of one permit a leaked slot is a process that answers "at capacity"
forever — no crash, no restart, just a task that looks healthy and busy.

The behavior itself is pinned in ``tests/infrastructure/test_turn_capacity.py``;
this guard stops a new caller from reintroducing the hand-rolled pairing.

``try_acquire`` belongs to the :class:`TurnGate` protocol alone, so it is an
offense wherever it is called — renaming the variable does not get a caller
past this. ``acquire`` / ``release`` are shared with locks and semaphores, so
those two count only on a gate-named receiver.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.shared.product_sources import product_python_files

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCT_ROOTS = (
    "bootstrap",
    "config",
    "core",
    "gateway",
    "integrations",
    "infrastructure",
    "surfaces",
    "tools",
)

#: The module that owns the two policies, and so owns the only acquire/release pair.
_SLOTS_MODULE = Path("infrastructure/process/turn_capacity/slots.py")

#: Unique to the :class:`TurnGate` protocol, so a call is a gate call whatever
#: the variable is called — an alias cannot hide behind a name lacking "gate".
_GATE_ONLY_METHOD = "try_acquire"

#: Shared with locks, semaphores and ``platform.release()``, so these count
#: only on a receiver that names a gate. Matching them everywhere would flag
#: unrelated code; the ``try_acquire`` rule above is what makes aliasing hard.
_AMBIGUOUS_METHODS = frozenset({"acquire", "release"})
_GATE_RECEIVER_TOKENS = ("gate", "permit", "slot")


def _is_test_path(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_")


def _gate_permit_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Permit calls that take a turn slot outside the capacity policies."""
    hits: list[tuple[int, str]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute):
                receiver = ast.unparse(func.value)
                named_like_a_gate = any(
                    token in receiver.lower() for token in _GATE_RECEIVER_TOKENS
                )
                if func.attr == _GATE_ONLY_METHOD or (
                    func.attr in _AMBIGUOUS_METHODS and named_like_a_gate
                ):
                    hits.append((node.lineno, f"{receiver}.{func.attr}()"))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


@pytest.mark.parametrize("package", _PRODUCT_ROOTS)
def test_turn_gate_permits_are_only_taken_inside_the_capacity_policies(package: str) -> None:
    root = _REPO_ROOT / package
    if not root.is_dir():
        pytest.skip(f"{package}/ missing")

    offenders: list[str] = []
    for path in product_python_files(root):
        if _is_test_path(path):
            continue
        relative = path.relative_to(_REPO_ROOT)
        if relative == _SLOTS_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, call in _gate_permit_calls(tree):
            offenders.append(f"{relative}:{lineno}: {call}")

    assert offenders == [], (
        "Take a turn slot through turn_slot() / queued_turn_slot() so the permit is "
        "released in a finally. A hand-paired acquire leaks the slot on any abnormal "
        "exit:\n" + "\n".join(offenders)
    )
