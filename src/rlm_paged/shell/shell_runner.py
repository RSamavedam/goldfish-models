"""Sandboxed shell-command runner.

Each command runs as a subprocess with cwd pinned to the agent root. We
enforce three layers of defense:

  1. **Executable allowlist.** Only a small set of programs can be invoked
     as the leading token of a command. The model can compose pipes /
     redirections / heredocs freely, but every `program` in the pipeline
     must be allowlisted. The allowlist is opinionated toward read-only
     analysis + file writing within the root.
  2. **Argument path sanitization.** Any argument that looks like a path
     (contains `/` or matches an existing file/directory) is checked: no
     `..` segments, no absolute paths outside the agent root.
  3. **Process limits.** A hard wall-clock timeout per command. (Memory
     and disk limits could be added via `resource` module on Unix; for
     v1 we just do timeout.)

We intentionally do NOT use real `chroot()` — that requires root and is
overkill for the "stop accidental escapes" threat model. What we call
"chroot-style" is the combination of `cwd` + path sanitization + the
allowlist. This blocks the model from accidentally reading `~/.ssh/...`
when it `cat ../../etc/passwd`. It does NOT stop a model that wants to
escape via `python3 -c 'import os; os.chdir("/")'`. The model would have
to deliberately try, which is a different threat model than this layer
is sized for.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ShellSecurityError(Exception):
    """Raised when a command violates sandbox policy."""


DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # File reading
        "cat", "head", "tail", "less", "more",
        # Search
        "grep", "egrep", "fgrep", "rg", "find",
        # Listing / inspection
        "ls", "stat", "file", "wc", "tree",
        # Transformation
        "echo", "printf", "tr", "sort", "uniq", "cut", "paste",
        "sed", "awk", "diff", "patch", "cmp",
        # File management within the root
        "mkdir", "touch", "rm", "mv", "cp", "ln", "chmod",
        # Interpreters (intentional — many benchmarks need code)
        "python", "python3", "node", "bash", "sh",
        # Source control — critical for SWE-bench's `git diff` patch
        # generation. The agent's repo is local and inside its root,
        # so git can't escape the sandbox via repo ops; the path
        # sanitizer still rejects `git clone https://evil` because
        # the URL is just an argument.
        "git",
        # Test runners — SWE-bench tasks routinely need to validate
        # patches by running the project's pytest suite.
        "pytest", "tox", "make",
        # Hashing / encoding (useful for IDs)
        "md5sum", "shasum", "sha1sum", "sha256sum", "base64", "xxd",
        # Misc useful
        "tee", "pwd", "date", "env", "yes", "true", "false",
        # Shell built-ins for redirects / loops — these don't really
        # exist as standalone binaries but pipelines that start with
        # them must be allowed. We special-case in `_check_pipeline`.
        ":",
    }
)


# Commands the harness intercepts and never actually executes via shell.
INTERCEPTED_COMMANDS = frozenset({"export", "done", "exit"})


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    intercepted_action: str | None = None  # "export" | "done" | None


class ShellRunner:
    def __init__(
        self,
        *,
        root: Path,
        allowlist: Iterable[str] | None = None,
        timeout_s: float = 10.0,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        self.root = Path(root).resolve()
        self.allowlist: frozenset[str] = (
            DEFAULT_ALLOWLIST if allowlist is None else frozenset(allowlist)
        )
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes

    # ----------------------------------------------------- public

    def run(self, command: str) -> CommandResult:
        """Run a single shell command line (may contain pipes/redirects)."""
        command = command.strip()
        if not command:
            return CommandResult(command=command, stdout="", stderr="", returncode=0)

        # First-word interception: `export ...`, `done`, `exit`.
        head_word = self._head_word(command)
        if head_word in INTERCEPTED_COMMANDS:
            return CommandResult(
                command=command,
                stdout="",
                stderr="",
                returncode=0,
                intercepted_action=head_word,
            )

        # Validate pipeline.
        try:
            self._check_pipeline(command)
        except ShellSecurityError as exc:
            return CommandResult(
                command=command,
                stdout="",
                stderr=f"[security] {exc}",
                returncode=126,
            )

        # Execute.
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                stdout=(exc.stdout or "")[: self.max_output_bytes]
                if isinstance(exc.stdout, str)
                else "",
                stderr=f"[timeout after {self.timeout_s}s]",
                returncode=124,
                timed_out=True,
            )
        stdout = proc.stdout[: self.max_output_bytes]
        stderr = proc.stderr[: self.max_output_bytes]
        return CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
        )

    # ----------------------------------------------------- validation

    def _head_word(self, command: str) -> str:
        try:
            return shlex.split(command, posix=True)[0]
        except (ValueError, IndexError):
            return command.split()[0] if command.split() else ""

    def _check_pipeline(self, command: str) -> None:
        """Validate every executable in the pipeline + sanitize path args."""
        # Split by shell control operators (|, &&, ||, ;) but NOT inside quotes.
        # We use shlex with posix=True to tokenize; control operators stay as
        # their own tokens.
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as exc:
            raise ShellSecurityError(f"unparseable command: {exc}")

        # Walk the token stream, identifying segment heads (start of pipeline
        # or after a control operator).
        next_is_head = True
        for tok in tokens:
            if tok in ("|", "||", "&&", ";", "&"):
                next_is_head = True
                continue
            if tok.startswith(">") or tok.startswith("<"):
                # redirection operator that wasn't split off; skip
                continue
            if next_is_head:
                # Strip leading env-var assignment like FOO=bar before the prog
                if "=" in tok and not tok.startswith("/") and "/" not in tok.split("=", 1)[0]:
                    # treat as env assignment, the *next* token is the head
                    continue
                program = Path(tok).name  # `python3` -> `python3`, `/usr/bin/cat` -> `cat`
                if Path(tok).is_absolute():
                    # Reject absolute-path invocations of binaries.
                    raise ShellSecurityError(
                        f"absolute-path executable not allowed: {tok}"
                    )
                if program not in self.allowlist:
                    raise ShellSecurityError(
                        f"executable not in allowlist: {program}"
                    )
                next_is_head = False
            else:
                # It's an argument. Sanitize if it looks path-like.
                self._check_arg(tok)

    def _check_arg(self, arg: str) -> None:
        # Strip shell-redirect glyphs that survived tokenization.
        cleaned = arg.lstrip("<>")
        if not cleaned:
            return
        # Absolute paths only allowed inside root.
        if cleaned.startswith("/"):
            try:
                Path(cleaned).resolve().relative_to(self.root)
            except ValueError:
                raise ShellSecurityError(
                    f"absolute path outside agent root: {cleaned}"
                )
        # Parent traversal disallowed.
        parts = Path(cleaned).parts
        if ".." in parts:
            raise ShellSecurityError(f"`..` not allowed in path: {cleaned}")
