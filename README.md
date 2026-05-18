# TTT-Discover Infrastructure

A modular Python skeleton for Test-Time Training to Discover (TTT-Discover)-style systems: online reinforcement learning at test time against a single hard problem with continuous verifiable reward.

This repo is intentionally a stub implementation. It gives you the interfaces, orchestration loop, buffer/search/trainer plumbing, example problems, configs, and unit tests needed to start swapping in real model backends and reward evaluators.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/run.py --config configs/default.yaml --budget 5
```

## What Works

- End-to-end generate -> evaluate -> train loop with no-op policy updates.
- In-memory solution buffer with uniform and PUCT-style selection.
- Entropic online trainer objective over stub training batches.
- Reward function registry and sandboxed Python execution helper.
- Toy sorting-network problem plus Triton matmul placeholder.
- JSONL logging and pickle-based checkpoint scaffolding.

## Roadmap

1. **Phase 1:** HF-backed generation, real training batches, toy problem loop.
2. **Phase 2:** PUCT selection, adaptive beta, sandboxed rewards, Triton kernels.
3. **Phase 3:** Async generation/evaluation/training, branching checkpoints, dashboards.
