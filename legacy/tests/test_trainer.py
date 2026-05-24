from ttt_discover.policy import Policy, TrainingBatch, TrainingExample
from ttt_discover.scheduler.beta import AdaptiveBetaScheduler
from ttt_discover.trainer import OnlineTrainer


def test_trainer_step_returns_entropic_metrics():
    policy = Policy()
    beta = AdaptiveBetaScheduler(1.0, "constant")
    trainer = OnlineTrainer(policy, beta, {})
    batch = TrainingBatch([TrainingExample("p", "c", reward=1.0, logprob=-0.5)])

    metrics = trainer.step(batch)

    assert metrics["loss"] > 0
    assert metrics["effective_beta"] == 1.0
    assert metrics["batch_size"] == 1.0
