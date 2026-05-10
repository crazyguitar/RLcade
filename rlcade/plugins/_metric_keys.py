"""Shared scalar key -> metric tag map used by metric-logging plugins."""

SCALAR_MAP: dict[str, str] = {
    "score": "reward/mean_score",
    "steps": "reward/total_steps",
    "policy_loss": "loss/policy",
    "value_loss": "loss/value",
    "entropy": "loss/entropy",
    "loss": "loss/total",
    "kl": "policy/kl",
    "clip_fraction": "policy/clip_fraction",
    "sps": "time/sps",
    "rollout": "time/rollout",
    "train": "time/train",
}
