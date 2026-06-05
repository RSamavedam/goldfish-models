# Goldfish-Models: filesystem-as-memory under aggressive context truncation

**Working title — refine later.** Alternates:
- "Notes from a goldfish: maintaining task state across a sliding context window"
- "Paged memory for stateless coding agents"

## One-sentence thesis

When you cap a frontier coding model's context window aggressively
(L = 128–2048 tokens), bare-prompt performance collapses through
characterizable failure modes (amnesia loops, thinking-cap collapses,
re-cat-instructions thrashing); a simple prompt protocol that makes
the model maintain a structured `notes.md` file across turns recovers
a non-trivial fraction of native-baseline solve rate.

## Why the paper exists

Long-horizon agents (Devin, OpenHands, Aider) all rely on the model
seeing its full conversation history every turn. As tasks get longer,
this hits an inference-cost wall. The natural escape — only show the
last *k* tokens of history — destroys the model's ability to track
state. We characterize *exactly how* this destruction happens, then
show that a filesystem-mediated scratchpad partially fixes it.

## Contributions

1. **A reproducible harness** (goldfish-models) that runs any chat
   model through SWE-bench Verified with a hard-capped per-turn
   context window. Open-sourced.
2. **Characterization of the goldfish regime.** With L=128–2048
   we see three named failure modes:
   - **Amnesia loops**: the model re-`cat`s `instructions.txt` 5+
     times per trajectory because the file's content is evicted
     from the window every few turns.
   - **Thinking-cap collapses**: reasoning models can emit zero
     output tokens for 4+ consecutive turns while their hidden
     reasoning hits the budget.
   - **Re-orient prose**: every turn opens with "I'll start by…",
     burning tokens that the next turn will discard.
3. **A protocol that helps.** A 1500-character "paged-memory
   protocol" added to the system prompt — mandating that every turn
   begin with `cat notes.md`, take one concrete action, and end with
   one appended line to `notes.md` — recovers Δ percentage points
   of solve rate at L=512/1024/2048, but is itself dominated by the
   window at L=128.
4. **A negative bound.** Below L≈X tokens, even the scratchpad
   protocol cannot recover any solve rate, because the act of
   re-reading `notes.md` itself consumes the entire window. There
   is a floor.

## Method

### Harness

- One trajectory ("cell") = (model, task, L, seed, prompt-variant).
- Each turn:
  1. The harness shows the model: a system prompt + the last L
     tokens of `history.txt` (the rolling terminal log).
  2. The model emits a response containing zero or more fenced
     `bash` blocks.
  3. The harness extracts every fenced block, runs the commands in
     order (chrooted to a per-cell agent directory), and appends
     STDOUT+STDERR to `history.txt`.
  4. Three intercepted commands: `export`, `done`, `exit`.
- Termination: model calls `done`/`exit`, hits max_turns=32, or
  hits cost_cap=250k tokens.

### Scoring

- SWE-bench Verified official Docker harness, run as subprocess
  per cell. Patch extracted from `user_output/answer.patch` if
  present, else `git diff` inside the agent's repo subdirectory.

### Prompt variants

- **Baseline.** Tells the model the rules (response format,
  allowlist of commands, hard rules about `export` before `done`).
  Does NOT mention `notes.md` or any scratchpad pattern.
- **Scratchpad.** Baseline + a "PAGED-MEMORY PROTOCOL" addendum
  (~1500 chars). Mandates:
  1. `cat notes.md` at the start of every turn.
  2. Take one concrete action.
  3. Append one line to `notes.md` describing what happened.
  Provides a template structure: `# Task`, `# Map`, `# Hypotheses`,
  `# Tried`, `# Next`.

### Sweep

- Provider: gpt-5
- Benchmark: SWE-bench Verified (`princeton-nlp/SWE-bench_Verified`)
- Tasks: astropy-12907, astropy-13033, astropy-13236,
  django-13551, sympy-13895 (5 tasks chosen for variety)
- L: 128, 512, 1024, 2048, ∞ (native baseline)
- Seeds: 1
- Variants: baseline + scratchpad
- Total cells: 50

## Headline result (placeholders; filled from paper sweep)

```
| L          | baseline | scratchpad | Δ (pp) |
|------------|----------|------------|--------|
| 128        | TBD      | TBD        | TBD    |
| 512        | TBD      | TBD        | TBD    |
| 1024       | TBD      | TBD        | TBD    |
| 2048       | TBD      | TBD        | TBD    |
| ∞ (native) | TBD      | TBD        | TBD    |
```

Figure 1: solve rate vs L, baseline vs scratchpad.

## Behavioral analysis (per-turn transcripts)

Independent of solve rate, we observe **how** the model fails:

- **Re-cat-instructions rate per cell vs L** — bare baseline goes
  from <1× at L=∞ to N× at L=512 (figure 4, "re-cat instructions"
  panel). Scratchpad reduces this because the model uses
  `notes.md` as the recovery anchor instead.
- **LENGTH-CAP rate vs L** — the reasoning model hits its
  per-turn output cap (with thinking tokens consuming the budget)
  at all L; not a goldfish-specific failure mode. We document it
  as a confound with a fix (bug 15: decouple `max_out` from L).

## Threats to validity / things to caveat

- **Single seed.** Solve-rate noise per cell is high. We report
  bands.
- **5 tasks ≠ 500.** We use a small task set because each cell at
  L=128 routinely takes 30+ turns. The trends should hold; the
  point estimates are noisy.
- **The scratchpad prompt was iterated** during early development.
  We did NOT optimize it against a held-out test set. The
  comparison is "ad-hoc prompt vs no prompt" — i.e., a sufficient
  but not necessarily tight upper bound on what prompt engineering
  can buy.
- **No human baseline** — we compare to native (L=∞) gpt-5 only.

## Sections (writing order)

1. **Abstract** (last)
2. **Introduction** — pitch the long-horizon agent inference-cost
   problem, name the goldfish regime, preview the figure.
3. **Method**
   - Harness architecture (1 fig: agent FS + turn pipeline)
   - Scoring
   - Prompt variants (verbatim in appendix)
4. **Results**
   - 4.1 Solve rate vs L (fig 1)
   - 4.2 Failure-mode breakdown (fig 3)
   - 4.3 Behavioral signatures (fig 4)
   - 4.4 Cost (fig 2)
5. **Discussion**
   - The scratchpad isn't magic; it's a way to spend window-tokens
     on the model's choice of content (notes) rather than the
     harness's choice (history.txt).
   - Lower bound: at L=128, `cat notes.md` output alone exceeds
     the window. There is no prompt that escapes this.
6. **Related work** — sliding-window attention, RAG, MemGPT,
   Reflexion, etc.
7. **Limitations / threats**
8. **Conclusion**
9. **Appendix A**: full prompts (baseline + scratchpad)
10. **Appendix B**: per-cell trajectory examples (best, worst,
    and one amnesia-loop)
11. **Appendix C**: bugs we found in our own harness during the
    investigation (worth its weight; "we almost wrote the wrong
    paper" honesty)

## Calls to action — what we need from the live runs

- **Cell results for all 25 baseline + 25 scratchpad cells.**
  Without these we have no headline.
- **For at least one solve at L<∞, the per-turn trace** that
  led to the solve. Goes in Appendix B.
- **For at least one amnesia loop, a trace of length ~15.**
  Appendix B + figure 4.
- **Token cost per cell, by L.** Powers the discussion section
  on inference-cost tradeoffs.

The runs in flight will produce all of the above. If the canary
shows that L=2048 baseline solves Astropy-12907 (which T6 looked
like), we should run the full paper sweep at the same matrix.

## Backup story if the scratchpad doesn't help

If the headline number shows NO improvement from scratchpad, the
paper pivots to **purely characterizing the goldfish regime** —
the failure modes are still novel, well-named, and reproducible.
We'd cut sections 4.4/5 about the protocol and lean harder on the
behavioral findings. Title change to "Characterizing context-window
collapse in agentic coding."

## What's not in this paper

- Training. No fine-tuning.
- Multi-model. Only gpt-5 in the headline table.
- Multi-benchmark. SWE-bench Verified only.
- Theory. Empirical only.
