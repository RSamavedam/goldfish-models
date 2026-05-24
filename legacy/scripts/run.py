from __future__ import annotations

import argparse

from ttt_discover.buffer import SolutionBuffer
from ttt_discover.engine import TTTEngine
from ttt_discover.policy import Policy
from ttt_discover.reward.registry import get_reward
from ttt_discover.scheduler.beta import AdaptiveBetaScheduler
from ttt_discover.trainer import OnlineTrainer
from ttt_discover.utils.config import load_config


def build_engine(config_path: str) -> TTTEngine:
    config = load_config(config_path)
    policy = Policy(
        backend=config.get("model", {}).get("backend", "huggingface"),
        model_name=config.get("model", {}).get("name"),
        max_tokens=config.get("model", {}).get("max_tokens"),
    )
    buffer = SolutionBuffer(max_size=config.get("buffer", {}).get("max_size"))
    beta = AdaptiveBetaScheduler(
        initial_beta=config.get("beta", {}).get("initial", 1.0),
        decay=config.get("beta", {}).get("schedule", "constant"),
        config=config.get("beta", {}),
    )
    trainer = OnlineTrainer(policy, beta, config.get("trainer", {}))
    reward_fn = get_reward(config.get("reward", {}).get("name", "sorting_net"))
    return TTTEngine(policy, buffer, trainer, reward_fn, None, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    engine = build_engine(args.config)
    budget = args.budget or int(engine.config.get("engine", {}).get("budget", 1))
    best = engine.run(budget)
    print(f"best_reward={best.reward:.4f}")
    print(best.text)


if __name__ == "__main__":
    main()
