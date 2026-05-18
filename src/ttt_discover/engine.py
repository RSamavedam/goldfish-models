from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttt_discover.buffer import Solution, SolutionBuffer
from ttt_discover.policy import Policy, TrainingBatch, TrainingExample
from ttt_discover.reward.base import RewardFunction


@dataclass
class StepResult:
    step: int
    candidates: list[Solution]
    best: Solution | None
    metrics: dict[str, Any] = field(default_factory=dict)


class TTTEngine:
    def __init__(
        self,
        policy: Policy,
        buffer: SolutionBuffer,
        trainer: Any,
        reward_fn: RewardFunction,
        search_strategy: Any,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.buffer = buffer
        self.trainer = trainer
        self.reward_fn = reward_fn
        self.search_strategy = search_strategy
        self.config = config
        self.step_idx = 0

    def run(self, budget: int) -> Solution:
        for _ in range(budget):
            self.step()
        best = self.best()
        if best is None:
            raise RuntimeError("TTT run produced no solutions")
        return best

    def step(self) -> StepResult:
        engine_cfg = self.config.get("engine", {})
        solutions_per_step = int(engine_cfg.get("solutions_per_step", 1))
        steps_per_update = int(engine_cfg.get("steps_per_update", 1))
        selection = self.config.get("buffer", {}).get("selection", "uniform")

        seeds = self._select_seeds(solutions_per_step, selection)
        prompts = [self._prompt_from_seed(seed) for seed in seeds]
        generations = self.policy.generate(prompts, temperature=1.0, n=1)

        candidates: list[Solution] = []
        for seed, generation in zip(seeds, generations, strict=False):
            reward = self.reward_fn(generation.text)
            solution = Solution(
                text=generation.text,
                reward=reward.reward,
                parent_id=seed.id if seed else None,
                step=self.step_idx,
                metadata={"reward": reward.metadata, "valid": reward.valid, "logprob": generation.logprob},
            )
            self.buffer.insert(solution)
            candidates.append(solution)

        metrics: dict[str, Any] = {"generated": len(candidates)}
        if steps_per_update > 0 and self.step_idx % steps_per_update == 0 and candidates:
            metrics.update(self.trainer.step(self._training_batch(candidates)))

        result = StepResult(step=self.step_idx, candidates=candidates, best=self.best(), metrics=metrics)
        self.step_idx += 1
        return result

    def best(self) -> Solution | None:
        best = self.buffer.best(k=1)
        return best[0] if best else None

    def _select_seeds(self, n: int, strategy: str) -> list[Solution | None]:
        if not self.buffer.all():
            return [None] * n
        if self.search_strategy is not None:
            return self.search_strategy.select(n)
        return self.buffer.select(n, strategy=strategy)

    def _prompt_from_seed(self, seed: Solution | None) -> str:
        if seed is None:
            return "Produce a candidate solution for the target problem."
        return f"Improve this candidate solution:\n\n{seed.text}"

    def _training_batch(self, solutions: list[Solution]) -> TrainingBatch:
        return TrainingBatch(
            examples=[
                TrainingExample(
                    prompt="",
                    completion=solution.text,
                    reward=solution.reward,
                    logprob=float(solution.metadata.get("logprob", 0.0)),
                    metadata={"solution_id": solution.id},
                )
                for solution in solutions
            ]
        )


def main() -> None:
    from ttt_discover.utils.config import load_config
    from ttt_discover.reward.registry import get_reward
    from ttt_discover.scheduler.beta import AdaptiveBetaScheduler
    from ttt_discover.trainer import OnlineTrainer

    config = load_config("configs/default.yaml")
    policy = Policy(**config.get("model", {}))
    buffer = SolutionBuffer(max_size=config.get("buffer", {}).get("max_size"))
    beta = AdaptiveBetaScheduler(config.get("beta", {}).get("initial", 1.0), "constant", config.get("beta", {}))
    trainer = OnlineTrainer(policy, beta, config.get("trainer", {}))
    engine = TTTEngine(policy, buffer, trainer, get_reward(config["reward"]["name"]), None, config)
    best = engine.run(int(config.get("engine", {}).get("budget", 1)))
    print(f"best reward={best.reward}: {best.text[:120]}")
