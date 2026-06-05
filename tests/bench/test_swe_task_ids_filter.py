"""SweBenchVerifiedSuite must support a task_ids filter so diagnostic
configs can pin a specific instance instead of taking whatever
happens to be first in dataset order.

`datasets` is not installed in the local dev env (only on EC2), so
we shim sys.modules['datasets'] with a fake load_dataset before
importing the suite.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

# Install a fake `datasets` module BEFORE the suite imports it.
_fake_datasets = types.ModuleType("datasets")
_fake_datasets.load_dataset = lambda *a, **k: []  # overridden per test
sys.modules.setdefault("datasets", _fake_datasets)

from rlm_paged.bench.swe_bench import SweBenchVerifiedSuite  # noqa: E402


_FAKE_ROWS = [
    {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy", "base_commit": "abc",
        "problem_statement": "x", "hints_text": None,
        "test_patch": "", "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]",
        "environment_setup_commit": None, "version": "1.0",
    },
    {
        "instance_id": "astropy__astropy-13033",
        "repo": "astropy/astropy", "base_commit": "def",
        "problem_statement": "y", "hints_text": None,
        "test_patch": "", "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]",
        "environment_setup_commit": None, "version": "1.0",
    },
    {
        "instance_id": "django__django-9999",
        "repo": "django/django", "base_commit": "ghi",
        "problem_statement": "z", "hints_text": None,
        "test_patch": "", "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]",
        "environment_setup_commit": None, "version": "1.0",
    },
]


def test_task_ids_filter_keeps_only_matching():
    with patch.object(sys.modules["datasets"], "load_dataset", return_value=_FAKE_ROWS):
        suite = SweBenchVerifiedSuite(task_ids=["astropy__astropy-12907"])
        tasks = suite.tasks()
    assert len(tasks) == 1
    assert tasks[0].task_id == "astropy__astropy-12907"


def test_task_ids_filter_with_multiple():
    with patch.object(sys.modules["datasets"], "load_dataset", return_value=_FAKE_ROWS):
        suite = SweBenchVerifiedSuite(
            task_ids=["astropy__astropy-12907", "django__django-9999"]
        )
        tasks = suite.tasks()
    assert {t.task_id for t in tasks} == {
        "astropy__astropy-12907", "django__django-9999"
    }


def test_no_task_ids_returns_everything_within_limit():
    with patch.object(sys.modules["datasets"], "load_dataset", return_value=_FAKE_ROWS):
        suite = SweBenchVerifiedSuite(limit=2)
        tasks = suite.tasks()
    assert len(tasks) == 2  # first two in dataset order


def test_task_ids_applied_before_limit():
    """The filter narrows first, then limit slices. If a config
    asks for one specific task, limit=1 shouldn't accidentally drop it."""
    with patch.object(sys.modules["datasets"], "load_dataset", return_value=_FAKE_ROWS):
        suite = SweBenchVerifiedSuite(
            task_ids=["django__django-9999"], limit=1
        )
        tasks = suite.tasks()
    assert len(tasks) == 1
    assert tasks[0].task_id == "django__django-9999"


def test_unknown_task_id_returns_empty():
    with patch.object(sys.modules["datasets"], "load_dataset", return_value=_FAKE_ROWS):
        suite = SweBenchVerifiedSuite(task_ids=["does-not-exist"])
        tasks = suite.tasks()
    assert tasks == []
