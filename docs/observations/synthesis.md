# Current-state synthesis

Rewritten on demand. This is what I believe NOW about model behavior
under the goldfish-models regime, with pointers into `journal.md`
for evidence. Stale claims get retired explicitly, not deleted.

**Last refreshed:** 2026-06-04 (pre-transcript scaffolding — no
per-turn data analyzed yet)

---

## Standing beliefs (with confidence)

### High confidence — backed by reproducible runs

- **gpt-5 + L=2048 + Astropy-12907 → solve** (smoke 4, smoke 5).
  This is currently the only end-to-end demonstration that the
  whole pipeline works (extraction → execution → patch → scoring).
- **Bug 12 (empty-done guard) catches a real failure mode.**
  Observed at least once on o4-mini smoke 5. Without the guard,
  the trajectory silently scores 0 despite the model showing intent.
- **`/tmp/` paths are essential for SWE-bench.** The tightened
  prompt now directs the model to `/tmp/answer.patch`. Sanitizer
  permissive prefix in both `agent_fs.py` and `shell_runner.py`.

### Medium confidence — single observations, plausible mechanism

- RLHF "finish and deliver" bias produces premature `done`s. We
  see refusals; we don't yet have enough data to know the rate.
- Under tight L, models spend a non-trivial fraction of turns
  re-orienting (re-`ls`, re-`git status`). Quantification waits
  for transcripts.
- Heredocs were a context-burn black hole pre-bug-7. Whether the
  fix fully removes that failure mode is unverified.

### Low confidence — speculative until we have turn data

- Wandering correlates with eventual solve (vs cost-cap death).
- "Confident-but-wrong" turns cluster on specific tasks (the
  model has a strong prior for the wrong fix).
- Empty-done refusals cluster on a small number of task_ids.

---

## Retired beliefs

- ~~"Bug 13 wandering terminator is needed."~~ Retired 2026-06-04
  per spec: the model has the right to wander/think even if it
  costs money. Bug 13 was reverted in commit `3436e92`.

---

## What we don't know yet

- Distribution of turn count at solve vs at cost-cap.
- Whether different providers (gpt-5 vs o4-mini vs claude) burn
  context in qualitatively different ways.
- How much of the L=512 failure rate is "not enough budget" vs
  "model can't do this task at any L".
- Whether `done REFUSED` actually changes subsequent behavior or
  the model just retries the same empty `done` with extra prose.

## Open questions to drive the next analysis pass

When fresh transcripts land, ask:

1. For each `cell_key`, count turns spent on (a) repo exploration,
   (b) code reading, (c) editing, (d) verification, (e) other.
2. Per-provider: distribution of `output_tokens` per turn.
   Hypothesis: gpt-5 emits longer turns than o4-mini.
3. How often does the SAME failing test get patched twice in one
   trajectory? Loop-detection signal.
4. Of refused `done`s: what does the response after the nag look
   like? Does the model recover, or does it nag-loop?
