"""Walled-off per-agent filesystem.

Each trajectory gets a private directory. The harness creates it at the
start, populates initial files, and tears it down (or archives it) at the
end. The model interacts with this directory exclusively via shell
commands.

Layout:

    <root>/
      instructions.txt        read-only; system instructions + task prompt
      stdin/                  per-turn stdin records (turn-N.txt)
      stdout/                 per-turn stdout records
      history.txt             merged terminal history (model output + cmd
                              transcripts), append-only by the harness;
                              the model can read it but never writes to it
                              directly
      user_output/            artifacts the model `export`ed for the user

The model can read/write any file *inside* this root via shell commands.
It cannot escape the root — that's enforced by `ShellRunner`. The four
top-level directories above are always created. The model is encouraged
(via instructions) to also create:

    continuing_instruction.txt   the model's note to its next turn
    notes/                       persistent knowledge it wants to keep
    knowledge_graph.txt          structured cross-references between notes

…but none of these is structurally required.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


AGENT_FILES = {
    "instructions": "instructions.txt",
    "history": "history.txt",
    "stdin_dir": "stdin",
    "stdout_dir": "stdout",
    "user_output_dir": "user_output",
}


@dataclass
class AgentFS:
    """The agent's private directory. Created and torn down by the harness."""

    root: Path
    instructions: str
    _created: bool = field(default=False, init=False)

    @classmethod
    def make(cls, root: str | Path, instructions: str) -> "AgentFS":
        fs = cls(root=Path(root).resolve(), instructions=instructions)
        fs.init()
        return fs

    def init(self) -> None:
        """Create the directory layout and populate initial files.

        Tolerates an already-existing root iff it's empty (lets us pair
        cleanly with tempfile.mkdtemp). Raises if the root has prior
        contents — we never want to mix a fresh agent into someone
        else's directory.
        """
        if self._created:
            return
        if self.root.exists():
            if any(self.root.iterdir()):
                raise FileExistsError(
                    f"agent root not empty: {self.root}"
                )
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        (self.root / AGENT_FILES["stdin_dir"]).mkdir()
        (self.root / AGENT_FILES["stdout_dir"]).mkdir()
        (self.root / AGENT_FILES["user_output_dir"]).mkdir()
        instr_path = self.root / AGENT_FILES["instructions"]
        instr_path.write_text(self.instructions, encoding="utf-8")
        instr_path.chmod(0o444)
        (self.root / AGENT_FILES["history"]).write_text("", encoding="utf-8")
        self._created = True

    # ----------------------------------------------------- accessors

    @property
    def instructions_path(self) -> Path:
        return self.root / AGENT_FILES["instructions"]

    @property
    def history_path(self) -> Path:
        return self.root / AGENT_FILES["history"]

    @property
    def user_output_dir(self) -> Path:
        return self.root / AGENT_FILES["user_output_dir"]

    def stdin_path(self, turn: int) -> Path:
        return self.root / AGENT_FILES["stdin_dir"] / f"turn-{turn:04d}.txt"

    def stdout_path(self, turn: int) -> Path:
        return self.root / AGENT_FILES["stdout_dir"] / f"turn-{turn:04d}.txt"

    # ----------------------------------------------------- writes

    def append_history(self, text: str) -> None:
        """Append a chunk to the merged terminal-history file."""
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def write_stdin(self, turn: int, text: str) -> None:
        self.stdin_path(turn).write_text(text, encoding="utf-8")

    def write_stdout(self, turn: int, text: str) -> None:
        self.stdout_path(turn).write_text(text, encoding="utf-8")

    def export(self, source: str, *, is_string: bool = False) -> Path:
        """Copy a file or string into `user_output/`. Returns the dest path.

        If `is_string` is True, `source` is the literal text to write. The
        destination filename is auto-generated.
        If False, `source` is a path *relative to the agent root*. Copies
        the file to user_output preserving its basename.

        Raises FileNotFoundError if the source is a path that doesn't exist.
        Raises ValueError on path-escape attempts.
        """
        if is_string:
            existing = list(self.user_output_dir.glob("export-*.txt"))
            n = len(existing)
            dest = self.user_output_dir / f"export-{n:04d}.txt"
            dest.write_text(source, encoding="utf-8")
            return dest
        src_path = self._resolve_export_source(source)
        if not src_path.exists():
            raise FileNotFoundError(f"export source not found: {source}")
        dest = self.user_output_dir / src_path.name
        # If a file with this name already exists, suffix it.
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while True:
                candidate = self.user_output_dir / f"{stem}-{i}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1
        shutil.copy2(src_path, dest)
        return dest

    # ----------------------------------------------------- reads

    def list_user_outputs(self) -> list[Path]:
        """Return all exported files, ordered by mtime ascending."""
        items = sorted(
            self.user_output_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
        )
        return [p for p in items if p.is_file()]

    def read_history_tail(self, max_chars: int) -> str:
        """Return the last `max_chars` bytes of history (rough approximation
        of last `k` tokens — caller does precise token-level truncation)."""
        path = self.history_path
        if not path.exists():
            return ""
        size = path.stat().st_size
        if size <= max_chars:
            return path.read_text(encoding="utf-8", errors="replace")
        with path.open("rb") as handle:
            handle.seek(size - max_chars)
            return handle.read().decode("utf-8", errors="replace")

    # ----------------------------------------------------- safety

    def _resolve_internal(self, relpath: str) -> Path:
        """Strict path resolution: must live inside the agent root.

        Raises ValueError if the resolved path escapes the root or contains
        path-escape attempts. Used for routine harness writes (history,
        stdin, stdout) where there's no legitimate reason to touch
        anything outside the root.
        """
        if relpath.startswith("/"):
            raise ValueError(f"absolute paths not allowed: {relpath!r}")
        if ".." in Path(relpath).parts:
            raise ValueError(f"parent-traversal not allowed: {relpath!r}")
        resolved = (self.root / relpath).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes agent root: {relpath!r}") from exc
        return resolved

    # Paths the model may legitimately export from. The agent root is
    # obviously allowed (model wrote a file inside its working dir). /tmp
    # is allowed because the model is encouraged to use /tmp as a scratch
    # space for in-flight artifacts like generated patches — see the
    # SWE-bench prompt's canonical pattern. Anything else is rejected.
    _EXPORT_ALLOWED_PREFIXES = ("/tmp/",)

    def _resolve_export_source(self, source: str) -> Path:
        """Permissive resolution used only by `export(is_string=False)`.

        Accepts:
          - Paths relative to the agent root (treated as inside-root).
          - Absolute paths starting with `/tmp/...` (process-private
            scratch).
        Rejects everything else with ValueError.
        """
        if source.startswith("/"):
            for allowed in self._EXPORT_ALLOWED_PREFIXES:
                if source.startswith(allowed):
                    p = Path(source).resolve()
                    # No `..` chicanery in /tmp either.
                    try:
                        p.relative_to(Path(allowed).resolve())
                    except ValueError as exc:
                        raise ValueError(
                            f"export path escapes {allowed}: {source!r}"
                        ) from exc
                    return p
            raise ValueError(
                f"absolute paths not allowed except under {self._EXPORT_ALLOWED_PREFIXES}: {source!r}"
            )
        if ".." in Path(source).parts:
            raise ValueError(f"parent-traversal not allowed: {source!r}")
        resolved = (self.root / source).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"path escapes agent root: {source!r}"
            ) from exc
        return resolved

    # ----------------------------------------------------- teardown

    def cleanup(self) -> None:
        """Remove the agent's directory tree. Idempotent."""
        if self.root.exists():
            # Restore instructions write permissions so shutil can remove it.
            try:
                self.instructions_path.chmod(0o644)
            except FileNotFoundError:
                pass
            shutil.rmtree(self.root, ignore_errors=True)
        self._created = False
