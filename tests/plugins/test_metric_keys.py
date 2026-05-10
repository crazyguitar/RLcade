"""SCALAR_MAP is the single source of truth for metric key -> tag mapping."""

from rlcade.plugins._metric_keys import SCALAR_MAP


def test_scalar_map_has_expected_keys():
    expected = {
        "score",
        "steps",
        "policy_loss",
        "value_loss",
        "entropy",
        "loss",
        "kl",
        "clip_fraction",
        "sps",
        "rollout",
        "train",
    }
    assert set(SCALAR_MAP.keys()) == expected


def test_scalar_map_tags_are_grouped():
    assert SCALAR_MAP["score"] == "reward/mean_score"
    assert SCALAR_MAP["loss"] == "loss/total"
    assert SCALAR_MAP["sps"] == "time/sps"


def test_tensorboard_uses_shared_map():
    from rlcade.plugins import tensorboard as tb_mod
    from rlcade.plugins._metric_keys import SCALAR_MAP

    assert tb_mod._SCALAR_MAP is SCALAR_MAP
