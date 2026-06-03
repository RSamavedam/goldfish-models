"""Tests for the SWE-bench loader + scorer that don't require Docker or HF gate.

The HF dataset load itself isn't tested here (network + gate); we test:

  - task_prompt rendering
  - patch sanity-check (`_looks_like_unified_diff`)
  - dry_run scorer always returns (False, 0.0)
  - extract_patch_from_agent_root prefers explicit user_output/*.patch
  - _maybe_load_json_list handles list / json-string / None
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlm_paged.bench.base import FAMILY_CODING, BenchTask
from rlm_paged.bench.swe_bench import (
    SweBenchVerifiedSuite,
    _looks_like_unified_diff,
    _maybe_load_json_list,
    extract_patch_from_agent_root,
)


def _make_task() -> BenchTask:
    return BenchTask(
        task_id="astropy__astropy-12345",
        family=FAMILY_CODING,
        payload={
            "instance_id": "astropy__astropy-12345",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "Bug: foo() returns None when bar.",
            "hints_text": None,
            "test_patch": "",
            "fail_to_pass": ["test_foo"],
            "pass_to_pass": ["test_baz"],
        },
        expected={
            "fail_to_pass": ["test_foo"],
            "pass_to_pass": ["test_baz"],
        },
    )


def test_task_prompt_includes_repo_commit_and_statement():
    suite = SweBenchVerifiedSuite(scorer_mode="dry_run")
    task = _make_task()
    prompt = suite.task_prompt(task)
    assert "astropy/astropy" in prompt
    assert "abc123" in prompt
    assert "foo() returns None" in prompt
    # Hints section absent when hints_text is None
    assert "DEVELOPER HINTS" not in prompt


def test_task_prompt_includes_hints_when_present():
    suite = SweBenchVerifiedSuite(scorer_mode="dry_run")
    task = _make_task()
    task.payload["hints_text"] = "Look at line 42 of utils.py."
    prompt = suite.task_prompt(task)
    assert "DEVELOPER HINTS" in prompt
    assert "line 42" in prompt


def test_looks_like_unified_diff_accepts_real_diff():
    diff = """--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
-def bar():
-    return None
+def bar():
+    return 42
"""
    assert _looks_like_unified_diff(diff)


def test_looks_like_unified_diff_rejects_garbage():
    assert not _looks_like_unified_diff("")
    assert not _looks_like_unified_diff("here is my answer: 42")
    assert not _looks_like_unified_diff("--- /dev/null")  # no +++ or @@


def test_dry_run_scorer_always_returns_false_zero():
    suite = SweBenchVerifiedSuite(scorer_mode="dry_run")
    task = _make_task()
    valid_diff = """--- a/x
+++ b/x
@@ -1 +1 @@
-a
+b
"""
    solved, score = suite.score(task, valid_diff)
    assert solved is False
    assert score == 0.0


def test_dry_run_scorer_rejects_non_diff():
    suite = SweBenchVerifiedSuite(scorer_mode="dry_run")
    task = _make_task()
    solved, score = suite.score(task, "not a patch")
    assert solved is False
    assert score == 0.0


def test_unknown_scorer_mode_raises():
    with pytest.raises(ValueError, match="unknown scorer_mode"):
        SweBenchVerifiedSuite(scorer_mode="moonshot")


def test_maybe_load_json_list_handles_real_list():
    assert _maybe_load_json_list(["a", "b"]) == ["a", "b"]


def test_maybe_load_json_list_parses_json_string():
    assert _maybe_load_json_list('["a", "b"]') == ["a", "b"]


def test_maybe_load_json_list_falls_back_on_invalid_json():
    assert _maybe_load_json_list("not json") == ["not json"]


def test_maybe_load_json_list_none_returns_empty():
    assert _maybe_load_json_list(None) == []


def test_extract_patch_prefers_explicit_user_output_file(tmp_path):
    root = tmp_path / "agent"
    (root / "user_output").mkdir(parents=True)
    (root / "user_output" / "answer.patch").write_text(
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    )
    patch = extract_patch_from_agent_root(root)
    assert "+b" in patch


def test_extract_patch_falls_back_to_any_patch_in_user_output(tmp_path):
    root = tmp_path / "agent"
    (root / "user_output").mkdir(parents=True)
    (root / "user_output" / "fix.patch").write_text(
        "--- a/y\n+++ b/y\n@@ -1 +1 @@\n-x\n+y\n"
    )
    patch = extract_patch_from_agent_root(root, explicit_patch_file=None)
    assert "+y" in patch


def test_extract_patch_falls_back_to_git_diff(tmp_path, monkeypatch):
    """If no exported patch exists, call git diff in the repo subdir."""
    import subprocess as sp

    root = tmp_path / "agent"
    repo = root / "repo"
    repo.mkdir(parents=True)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        out = sp.CompletedProcess(
            args=cmd, returncode=0,
            stdout="--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n",
            stderr="",
        )
        return out

    monkeypatch.setattr("rlm_paged.bench.swe_bench.subprocess.run", fake_run)
    patch = extract_patch_from_agent_root(root, explicit_patch_file=None)
    assert "+new" in patch
    assert captured["cmd"][0] == "git"
    assert captured["cwd"] == repo


def test_extract_patch_empty_when_nothing_to_diff(tmp_path):
    root = tmp_path / "agent"
    root.mkdir()
    # no repo, no user_output, nothing
    assert extract_patch_from_agent_root(root, explicit_patch_file=None) == ""
