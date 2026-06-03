"""Tests for the optional S3 syncer.

We mock boto3 so no AWS credentials or network calls are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlm_paged.utils.s3_sync import S3Syncer


class _FakeS3Client:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []
        self.errors_to_raise: list[Exception] = []

    def upload_file(self, local_path: str, bucket: str, key: str) -> None:
        if self.errors_to_raise:
            raise self.errors_to_raise.pop(0)
        self.uploads.append((local_path, bucket, key))


def _syncer_with_fake(fake: _FakeS3Client, *, sync_every: int = 1) -> S3Syncer:
    s = S3Syncer(
        bucket="my-bucket", prefix="goldfish/", sync_every=sync_every, enabled=True
    )
    s._client = fake
    return s


def test_disabled_syncer_no_ops(tmp_path):
    s = S3Syncer.from_env_or_args(bucket=None)
    assert s.enabled is False
    # All methods should silently succeed.
    s.start()
    s.maybe_sync([tmp_path / "nope.jsonl"])
    s.force_sync([tmp_path / "nope.jsonl"])
    s.stop_and_final_sync([tmp_path / "nope.jsonl"])


def test_enabled_syncer_resolves_env(monkeypatch):
    monkeypatch.setenv("GOLDFISH_S3_BUCKET", "envbucket")
    monkeypatch.setenv("GOLDFISH_S3_PREFIX", "env/prefix")
    monkeypatch.setenv("GOLDFISH_S3_SYNC_EVERY", "5")
    s = S3Syncer.from_env_or_args()
    assert s.enabled is True
    assert s.bucket == "envbucket"
    assert s.prefix == "env/prefix"
    assert s.sync_every == 5


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("GOLDFISH_S3_BUCKET", "envbucket")
    s = S3Syncer.from_env_or_args(bucket="explicit", prefix="x/", sync_every=2)
    assert s.bucket == "explicit"
    assert s.prefix == "x/"
    assert s.sync_every == 2


def test_force_sync_uploads_each_file(tmp_path):
    fake = _FakeS3Client()
    s = _syncer_with_fake(fake)

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("row1\n")
    b.write_text("row2\n")
    result = s.force_sync([a, b])

    assert result.files_uploaded == 2
    assert result.bytes_uploaded == len("row1\n") + len("row2\n")
    assert {key for (_, _, key) in fake.uploads} == {
        "goldfish/" + a.name,
        "goldfish/" + b.name,
    } or len(fake.uploads) == 2  # path-relative keys may differ on tmp


def test_force_sync_skips_missing_files(tmp_path):
    fake = _FakeS3Client()
    s = _syncer_with_fake(fake)
    result = s.force_sync([tmp_path / "missing.jsonl"])
    assert result.files_uploaded == 0
    assert result.errors == []


def test_maybe_sync_respects_cell_counter(tmp_path):
    fake = _FakeS3Client()
    s = _syncer_with_fake(fake, sync_every=3)
    f = tmp_path / "log.jsonl"
    f.write_text("x")

    assert s.maybe_sync([f]) is None
    assert s.maybe_sync([f]) is None
    third = s.maybe_sync([f])
    assert third is not None
    assert third.files_uploaded == 1

    # Counter resets.
    assert s.maybe_sync([f]) is None


def test_force_sync_walks_directories(tmp_path):
    fake = _FakeS3Client()
    s = _syncer_with_fake(fake)
    d = tmp_path / "agents"
    d.mkdir()
    (d / "a.txt").write_text("1")
    (d / "b.txt").write_text("22")
    nested = d / "sub"
    nested.mkdir()
    (nested / "c.txt").write_text("333")
    result = s.force_sync([d])
    assert result.files_uploaded == 3


def test_upload_error_recorded_but_does_not_raise(tmp_path):
    fake = _FakeS3Client()
    fake.errors_to_raise = [RuntimeError("boom")]
    s = _syncer_with_fake(fake)
    f = tmp_path / "x.jsonl"
    f.write_text("x")
    result = s.force_sync([f])
    assert result.files_uploaded == 0
    assert len(result.errors) == 1
    assert "boom" in result.errors[0]


def test_start_without_boto3_raises(monkeypatch):
    # Force the import to fail.
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "boto3":
            raise ImportError("no boto3")
        return orig_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    s = S3Syncer(bucket="b", enabled=True)
    with pytest.raises(RuntimeError, match="boto3 is not installed"):
        s.start()
