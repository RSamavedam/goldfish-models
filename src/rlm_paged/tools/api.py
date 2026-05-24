from __future__ import annotations

from dataclasses import dataclass

from rlm_paged.store.store import ChunkStore
from rlm_paged.window.state import ActiveWindow, WindowViolation


@dataclass
class OpResult:
    ok: bool
    payload: object = None
    error: str | None = None


@dataclass
class ParsedOp:
    code: str
    args: tuple[str, ...]


def parse_op(line: str) -> ParsedOp | None:
    """Parse one line of model output into a ParsedOp, or None if not an op.

    A line is an op iff it starts with a known single-char op code followed
    by whitespace and at least one argument. Everything else is treated as
    free-form reasoning that lands in the tail.
    """
    stripped = line.strip()
    if len(stripped) < 3:
        return None
    code = stripped[0]
    if code not in {"e", "r", "q", "a", "l", "s"}:
        return None
    if not stripped[1].isspace():
        return None
    args = tuple(stripped[2:].split())
    if not args:
        return None
    return ParsedOp(code=code, args=args)


class ToolDispatcher:
    """Executes parsed ops against the (window, store) pair.

    The dispatcher is intentionally simple: it does not generate tokens, it
    does not call the model. The harness loop calls it once per parsed op
    and handles the returned OpResult.
    """

    def __init__(self, window: ActiveWindow, store: ChunkStore, *, step: int = 0) -> None:
        self.window = window
        self.store = store
        self.step = step

    def dispatch(self, op: ParsedOp) -> OpResult:
        try:
            handler = getattr(self, f"_op_{op.code}")
        except AttributeError:
            return OpResult(False, error=f"unknown op: {op.code}")
        try:
            return handler(op.args)
        except (ValueError, IndexError, KeyError, WindowViolation) as exc:
            return OpResult(False, error=f"{type(exc).__name__}: {exc}")

    # -- op implementations --

    def _op_e(self, args: tuple[str, ...]) -> OpResult:
        (n_str,) = args[:1]
        freed = self.window.evict_head(int(n_str))
        return OpResult(True, payload={"freed": freed})

    def _op_r(self, args: tuple[str, ...]) -> OpResult:
        cid_str, ofs_str, len_str = args[:3]
        cid, ofs, length = int(cid_str), int(ofs_str), int(len_str)
        if not self.window.can_fit(length):
            return OpResult(False, error="window full; evict before retrieving")
        tokens = self.store.retrieve(cid, ofs, length, at_step=self.step)
        self.window.append_tail(length)
        return OpResult(True, payload={"tokens": tokens, "chunk_id": cid})

    def _op_q(self, args: tuple[str, ...]) -> OpResult:
        (cid_str,) = args[:1]
        out, inc = self.store.refs(int(cid_str))
        return OpResult(True, payload={"out": out, "in": inc})

    def _op_a(self, args: tuple[str, ...]) -> OpResult:
        cid_str, tag = args[0], " ".join(args[1:])[:32]
        self.store.annotate(int(cid_str), tag)
        return OpResult(True)

    def _op_l(self, args: tuple[str, ...]) -> OpResult:
        a_str, b_str = args[:2]
        self.store.link(int(a_str), int(b_str))
        return OpResult(True)

    def _op_s(self, args: tuple[str, ...]) -> OpResult:
        return OpResult(False, error="similarity search not implemented (Phase 1.5)")
