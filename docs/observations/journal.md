# Behavior journal

Append-only chronological log. Each entry: what was looked at, what
stood out, hypotheses, predictions. Older entries are NOT edited;
they're the historical record.

Entry format:

    ## YYYY-MM-DD — short title
    **Source:** runs/turns_xxx.jsonl (cells m..n)
    **What I looked at:** ...
    **Observations:** ...
    **Hypotheses:** ...
    **Predictions / next checks:** ...

---

## 2026-06-04 — Logger online, baseline established before transcripts

**Source:** Pre-transcript era. Smoke 1–6 ran against `runs/smoke*.jsonl`
without per-turn capture. This entry records what was already known
from inspecting agent directories live during smoke 4–6.

**Observations (carried forward):**

- **gpt-5 reliably solves Astropy-12907** at L=2048 across smoke 4 + 5
  once bugs 4–10 were fixed. The fix in `argparse.py` (`required=False`
  on the `_actions_taken` argument) survived the Docker-based scorer.
- **o4-mini delivered an empty `done`** on smoke 5 — caught by bug-12
  guard which now refuses the terminate and writes `[done REFUSED]`
  back into history. Without the guard this trajectory would have
  scored solved=False despite the model showing intent to solve.
- **Heredoc rejection (bug 7)** was producing a feedback loop where
  the model would write a multi-line patch, see "command not allowed",
  and try increasingly elaborate workarounds — burning context on
  the bug rather than the task.
- **`cd` failures (bug 9)** appeared as the model "forgetting" its
  cwd between turns. It wasn't forgetting — every `cd` was being
  rejected silently.

**Hypotheses (open):**

1. RLHF makes models bias toward "finish and deliver something",
   producing premature empty `done`s. The harness's `done REFUSED`
   nag may need stronger wording to override the prior.
2. Under tight L, models burn budget describing what they're about
   to do instead of doing it. A "no preamble" prompt rule may help.
3. Wandering (4+ empty turns) is sometimes legitimate
   reflection — we will NOT terminate it, per spec, and instead
   collect data on whether wandering correlates with eventual
   solve or with cost-cap death.

**Predictions to check once transcripts land:**

- Empty-done refusals will cluster on a small number of task_ids —
  not be uniformly distributed.
- Most failed solves will have at least one turn where the model
  output looks confident (no hedging, no "let me try") but executes
  the wrong fix.
- L=512 and L=1024 trajectories will spend >30% of turns
  re-establishing context (re-`ls`, re-`cat`, re-`git status`).

---

<!-- Append new entries above this line as they happen. -->
