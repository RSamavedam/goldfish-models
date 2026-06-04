"""SWE-bench Verified loader + Docker-based scorer.

Dataset: `princeton-nlp/SWE-bench_Verified` (HF-gated; the user must accept
the dataset terms on HuggingFace and have `HF_TOKEN` set on the host
running the loader).

Each BenchTask carries the full SWE-bench instance in `payload`:

    payload = {
        "instance_id":       str,        # e.g. "astropy__astropy-12345"
        "repo":              str,        # e.g. "astropy/astropy"
        "base_commit":       str,        # the SHA the model starts from
        "problem_statement": str,        # the issue text the model reads
        "hints_text":        str | None,
        "test_patch":        str,        # the harness applies this BEFORE
                                         # running tests, NOT before model work
        "fail_to_pass":      list[str],  # tests that must pass after patch
        "pass_to_pass":      list[str],  # tests that must keep passing
        "environment_setup_commit": str | None,
        "version":           str | None, # repo version label
    }

The scorer expects `response` to be a unified-diff string (the model's
patch). Patch extraction from the agent's filesystem is the caller's
responsibility — `score()` is pure: patch in, pass/fail out.

Two scorer modes:

  - `subprocess` (default): runs the official `swebench.harness.run_evaluation`
    in a Docker sidecar against this instance, captures the test report,
    returns its verdict. Requires Docker on the host and the `swebench`
    Python package installed in the *host's* environment (not the agent's).
    This is the canonical, slow, correct path.

  - `dry_run`: skips Docker entirely; just checks that the diff parses as
    a valid unified-diff and contains at least one hunk. Use for tests
    and for local smoke runs where you don't want to spin up Docker.
    Never produces a `solved=True` verdict; always `(False, 0.0)`.

The mode is set per-suite at construction. The cloud sweep uses
`subprocess`; tests and laptop runs default to `dry_run`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rlm_paged.bench.base import FAMILY_CODING, BenchSuite, BenchTask


DATASET_ID = "princeton-nlp/SWE-bench_Verified"

SWE_BENCH_TASK_PROMPT_TEMPLATE = """You are working on a real software bug.

REPOSITORY: {repo}
COMMIT:     {base_commit}

The repository is checked out at the above commit in your agent root,
under `./repo/`.

WHAT YOU MUST DELIVER
=====================
The scorer expects ONE thing: a unified-diff patch in user_output/.
The canonical way to produce it:

    ```bash
    # 1. Make your edits inside ./repo/ (using sed, python, or by
    #    `cat > file <<EOF` heredocs — any way to mutate the source).

    # 2. When you believe the fix is right, generate the patch:
    git -C repo diff > /tmp/answer.patch

    # 3. Sanity-check the diff is non-empty and points at the files
    #    you actually changed:
    wc -l /tmp/answer.patch
    head -20 /tmp/answer.patch

    # 4. Export and finish:
    export /tmp/answer.patch
    done
    ```

The scorer runs the project's hidden test suite against this patch.
NO patch in user_output/ means the score is 0 even if you understood
the bug perfectly. DO NOT use `export-string` with a description like
"the diff above" — the scorer will read those literal words as your
patch and reject it.

PROBLEM STATEMENT
=================
{problem_statement}

{hints_section}

GROUND-TRUTH TESTS
==================
The harness will run a held-out set of tests we are NOT showing you.
Some currently fail; your patch must make them pass without breaking
tests that currently pass. The problem statement above describes those
tests informally.

USEFUL EXPLORATION COMMANDS
===========================
  - `ls -la repo/` to see the top-level layout
  - `cat repo/README*` or `cat repo/docs/index.*` for orientation
  - `cat repo/CONTRIBUTING*` for testing conventions
  - `git -C repo log --oneline -20` to see recent activity
  - `grep -rn 'symbol_name' repo/src/` to find relevant code

You can run tests with `pytest` inside `./repo/` if the project uses
pytest. Keep iterations small; the cost cap is real.
"""


class SweBenchVerifiedSuite(BenchSuite):
    """SWE-bench Verified subset.

    Construction:
        suite = SweBenchVerifiedSuite(limit=50, scorer_mode="dry_run")
        tasks = suite.tasks()           # loads the HF dataset
        suite.score(task, patch_text)   # runs the test suite via Docker
    """

    def __init__(
        self,
        *,
        split: str = "test",
        limit: int | None = None,
        scorer_mode: str = "dry_run",
        scorer_timeout_s: float = 1800.0,
        docker_image_override: str | None = None,
    ) -> None:
        if scorer_mode not in {"dry_run", "subprocess"}:
            raise ValueError(f"unknown scorer_mode: {scorer_mode!r}")
        self.split = split
        self.limit = limit
        self.scorer_mode = scorer_mode
        self.scorer_timeout_s = scorer_timeout_s
        self.docker_image_override = docker_image_override
        self._tasks: list[BenchTask] | None = None

    @property
    def family(self) -> str:
        return FAMILY_CODING

    @property
    def name(self) -> str:
        return "swe_bench_verified"

    def tasks(self) -> list[BenchTask]:
        if self._tasks is not None:
            return self._tasks
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset(DATASET_ID, split=self.split)
        out: list[BenchTask] = []
        for row in ds:
            payload = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "hints_text": row.get("hints_text"),
                "test_patch": row["test_patch"],
                "fail_to_pass": _maybe_load_json_list(row.get("FAIL_TO_PASS")),
                "pass_to_pass": _maybe_load_json_list(row.get("PASS_TO_PASS")),
                "environment_setup_commit": row.get("environment_setup_commit"),
                "version": row.get("version"),
            }
            out.append(
                BenchTask(
                    task_id=row["instance_id"],
                    family=FAMILY_CODING,
                    payload=payload,
                    expected={
                        "fail_to_pass": payload["fail_to_pass"],
                        "pass_to_pass": payload["pass_to_pass"],
                    },
                )
            )
        if self.limit:
            out = out[: self.limit]
        self._tasks = out
        return out

    def task_prompt(self, task: BenchTask) -> str:
        p = task.payload
        hints = p.get("hints_text")
        hints_section = (
            f"DEVELOPER HINTS\n===============\n{hints}\n" if hints else ""
        )
        return SWE_BENCH_TASK_PROMPT_TEMPLATE.format(
            repo=p["repo"],
            base_commit=p["base_commit"],
            problem_statement=p["problem_statement"],
            hints_section=hints_section,
        )

    # ----------------------------------------------------- scoring

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        """Score a patch against the SWE-bench verdict.

        `response` is interpreted as a unified-diff patch. If it doesn't
        parse, the task is graded `(False, 0.0)`. Otherwise we dispatch
        to the configured scorer mode.
        """
        if not _looks_like_unified_diff(response):
            return False, 0.0

        if self.scorer_mode == "dry_run":
            # Patch parses; we DON'T claim it's correct.
            return False, 0.0
        if self.scorer_mode == "subprocess":
            return self._score_via_subprocess(task, response)
        raise RuntimeError(f"unreachable scorer_mode: {self.scorer_mode}")

    def _score_via_subprocess(
        self, task: BenchTask, patch_text: str
    ) -> tuple[bool, float]:
        """Run the official SWE-bench evaluation harness in a sidecar Docker.

        Implementation note: we use the `swebench` PyPI package's CLI
        rather than importing it as a library because the package version
        + the per-instance Docker images are tightly coupled and we want
        the official entry point's behavior verbatim.

        Returns (solved, score). On any error we return (False, 0.0) and
        store the error in a sibling JSON file for postmortem.
        """
        instance_id = task.payload["instance_id"]
        with tempfile.TemporaryDirectory(prefix=f"swe-eval-{instance_id}-") as tmpdir:
            tmp = Path(tmpdir)
            patch_path = tmp / "model.patch"
            patch_path.write_text(patch_text, encoding="utf-8")

            # The harness wants a JSONL of predictions, one per instance.
            predictions_path = tmp / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": "goldfish-models",
                        "model_patch": patch_text,
                    }
                ) + "\n",
                encoding="utf-8",
            )

            log_dir = tmp / "logs"
            log_dir.mkdir()
            # Use sys.executable, not the bare string "python" — Amazon
            # Linux 2023 (and many minimal distros) ship `python3` only,
            # no `python` symlink, and bare "python" raises
            # FileNotFoundError. The exception handler below used to
            # silently treat that as "swebench not installed", which is
            # how the first cloud sweep scored every patch False.
            cmd = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--predictions_path", str(predictions_path),
                "--max_workers", "1",
                "--run_id", f"goldfish-{instance_id}",
                "--instance_ids", instance_id,
                "--dataset_name", DATASET_ID,
                "--split", self.split,
                "--cache_level", "instance",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.scorer_timeout_s,
                )
            except subprocess.TimeoutExpired:
                # Write a sibling debug file so a stuck scorer is
                # diagnosable from the JSONL rerun later.
                (tmp / "_scorer_timeout").write_text(
                    f"timeout after {self.scorer_timeout_s}s\n"
                )
                return False, 0.0
            except FileNotFoundError as exc:
                # Almost certainly swebench is genuinely missing — but
                # we no longer hide it, since the bare "python" mistake
                # would mask real environment failures.
                (tmp / "_scorer_not_installed").write_text(
                    f"FileNotFoundError: {exc}\n"
                )
                return False, 0.0

            verdict, score = _read_swebench_report(log_dir, instance_id)
            if verdict is None:
                # Look in stdout for the rendered summary as a fallback.
                verdict, score = _parse_summary_from_stdout(
                    proc.stdout + proc.stderr, instance_id
                )
            return bool(verdict), float(score)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _maybe_load_json_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [str(x) for x in json.loads(raw)]
        except json.JSONDecodeError:
            return [raw]
    return [str(raw)]


def _looks_like_unified_diff(text: str) -> bool:
    """Cheap structural check that `text` is a unified diff.

    Requires at least one `--- ` / `+++ ` file-header pair and one `@@`
    hunk header. Tolerates surrounding prose (the model might prefix
    the diff with explanation).
    """
    if not text:
        return False
    has_minus = "\n--- " in ("\n" + text)
    has_plus = "\n+++ " in ("\n" + text)
    has_hunk = "\n@@" in ("\n" + text)
    return has_minus and has_plus and has_hunk


def _read_swebench_report(
    log_dir: Path, instance_id: str
) -> tuple[bool | None, float]:
    """Find and parse the per-instance report.json the harness writes.

    Returns (verdict_or_None, score_in_[0,1]). `verdict_or_None` is None
    when no report can be located — the caller falls back to stdout
    parsing.
    """
    candidates = list(log_dir.rglob("report.json")) + list(
        log_dir.rglob(f"{instance_id}*.json")
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # The harness's report.json has shape:
        # {instance_id: {"resolved": bool, "tests_status": {...}, ...}}
        if isinstance(data, dict) and instance_id in data:
            inst = data[instance_id]
            resolved = bool(inst.get("resolved", False))
            return resolved, 1.0 if resolved else 0.0
        if isinstance(data, dict) and "resolved" in data:
            resolved = bool(data["resolved"])
            return resolved, 1.0 if resolved else 0.0
    return None, 0.0


def _parse_summary_from_stdout(
    output: str, instance_id: str
) -> tuple[bool, float]:
    """Last-ditch parser if the JSON report wasn't found."""
    lowered = output.lower()
    if "resolved instances: 1" in lowered:
        return True, 1.0
    if "resolved instances: 0" in lowered:
        return False, 0.0
    return False, 0.0


# --------------------------------------------------------------------- #
# Patch extraction from the agent's filesystem                          #
# --------------------------------------------------------------------- #


def extract_patch_from_agent_root(
    agent_root: Path,
    *,
    repo_subdir: str = "repo",
    explicit_patch_file: str | None = "answer.patch",
) -> str:
    """Build a unified-diff patch from the agent's working repo.

    Priority order:
      1. If `explicit_patch_file` exists in user_output/, use its
         contents verbatim. (The model exported a patch directly.)
      2. If a file with the same name exists at the top level of the
         agent root, use that.
      3. Otherwise run `git diff` inside `repo_subdir` and return the
         result.

    Returns "" if none of those produce content.
    """
    if explicit_patch_file:
        for candidate in (
            agent_root / "user_output" / explicit_patch_file,
            agent_root / explicit_patch_file,
        ):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate.read_text(encoding="utf-8", errors="replace")

    # Also accept any *.patch in user_output as the answer.
    user_out = agent_root / "user_output"
    if user_out.is_dir():
        patches = sorted(user_out.glob("*.patch"))
        if patches:
            return patches[-1].read_text(encoding="utf-8", errors="replace")

    # Fall back to git diff.
    repo_path = agent_root / repo_subdir
    if not repo_path.exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout or ""


def bootstrap_repo_into_agent_root(
    agent_root: Path,
    *,
    repo: str,
    base_commit: str,
    repo_subdir: str = "repo",
    cache_dir: Path | None = None,
) -> None:
    """Check out `repo` at `base_commit` under `agent_root/repo_subdir`.

    Strategy:
      1. If `cache_dir / repo_slug` exists, use it as a local source
         (`git clone --shared` for speed); otherwise clone from GitHub.
      2. Hard reset to `base_commit`.
      3. Clean.

    Raises subprocess.CalledProcessError on failure.
    """
    dest = agent_root / repo_subdir
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    repo_slug = repo.replace("/", "__")
    if cache_dir is not None and (cache_dir / repo_slug).exists():
        clone_src = str(cache_dir / repo_slug)
        subprocess.run(
            ["git", "clone", "--shared", clone_src, str(dest)],
            check=True, capture_output=True, text=True,
        )
    else:
        url = f"https://github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", url, str(dest)],
            check=True, capture_output=True, text=True,
        )
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=dest, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "clean", "-fdx"],
        cwd=dest, check=True, capture_output=True, text=True,
    )
