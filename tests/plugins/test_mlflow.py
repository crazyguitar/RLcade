"""MLflowPlugin tests -- use a local file:// tracking URI in tmp_path."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from rlcade.plugins import TrainerPlugin
from rlcade.plugins.mlflow import MLflowPlugin


def _make_args(**overrides) -> Namespace:
    base = dict(
        agent="ppo",
        max_iterations=1,
        checkpoint_path="ckpt.pt",
        safetensors_path="",
        mlflow="",
        mlflow_experiment="rlcade",
        mlflow_run_name=None,
    )
    base.update(overrides)
    return Namespace(**base)


def _make_plugin(tmp_path: Path, **kwargs) -> MLflowPlugin:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    args = kwargs.pop("args", _make_args())
    return MLflowPlugin(
        tracking_uri=kwargs.pop("tracking_uri", tracking_uri),
        experiment=kwargs.pop("experiment", "rlcade-test"),
        run_name=kwargs.pop("run_name", None),
        checkpoint_path=kwargs.pop("checkpoint_path", str(tmp_path / "ckpt.pt")),
        safetensors_path=kwargs.pop("safetensors_path", ""),
        args=args,
    )


def test_satisfies_trainer_plugin_protocol(tmp_path):
    plugin = _make_plugin(tmp_path)
    assert isinstance(plugin, TrainerPlugin)


def test_constructor_does_not_touch_mlflow(tmp_path, monkeypatch):
    """Construction must not call set_tracking_uri or start_run."""
    calls = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.append(("uri", uri)))
    monkeypatch.setattr(mlflow, "start_run", lambda **k: calls.append(("start_run", k)))
    _make_plugin(tmp_path)
    assert calls == []


def _runs_in(tmp_path):
    """Return all MLflow runs under tmp_path's tracking dir."""
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    return mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")


def test_on_setup_starts_run_and_logs_params(tmp_path):
    args = _make_args(agent="ppo", max_iterations=42, checkpoint_path="x.pt")
    plugin = _make_plugin(tmp_path, args=args)

    plugin.on_setup(trainer=None)
    mlflow.end_run()

    runs = _runs_in(tmp_path)
    assert len(runs) == 1
    params = runs[0].data.params
    assert params["agent"] == "ppo"
    assert params["max_iterations"] == "42"
    assert params["checkpoint_path"] == "x.pt"


def test_on_setup_truncates_long_param_values(tmp_path):
    long_value = "x" * 2000
    args = _make_args()
    args.long_thing = long_value
    plugin = _make_plugin(tmp_path, args=args)

    plugin.on_setup(trainer=None)
    mlflow.end_run()

    runs = _runs_in(tmp_path)
    assert len(runs[0].data.params["long_thing"]) <= 500


def test_on_setup_skips_underscore_keys(tmp_path):
    args = _make_args()
    args._private = "should-not-appear"
    plugin = _make_plugin(tmp_path, args=args)

    plugin.on_setup(trainer=None)
    mlflow.end_run()

    runs = _runs_in(tmp_path)
    assert "_private" not in runs[0].data.params


def _metric_history(tmp_path, metric_name):
    """Return [(step, value), ...] for the given metric of the most-recent run."""
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    assert runs, "no runs found"
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tmp_path / 'mlruns'}")
    history = client.get_metric_history(runs[0].info.run_id, metric_name)
    return [(m.step, m.value) for m in history]


def test_on_step_end_logs_known_keys(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.on_setup(trainer=None)

    plugin.on_step_end(
        trainer=None,
        iteration=5,
        summary={
            "score": 12.5,
            "loss": 0.75,
            "unknown_thing": 99,
        },
    )

    plugin.on_done(trainer=None)
    mlflow.end_run()

    assert _metric_history(tmp_path, "reward/mean_score") == [(5, 12.5)]
    assert _metric_history(tmp_path, "loss/total") == [(5, 0.75)]
    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    metric_keys = set(runs[0].data.metrics.keys())
    assert "unknown_thing" not in metric_keys


def test_on_step_end_ignores_empty_summary(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.on_setup(trainer=None)
    plugin.on_step_end(trainer=None, iteration=1, summary=None)
    plugin.on_step_end(trainer=None, iteration=2, summary={})
    plugin.on_done(trainer=None)
    mlflow.end_run()

    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    assert runs[0].data.metrics == {}


def test_evaluation_hook_logs_three_metrics(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.on_setup(trainer=None)

    plugin.on_eval(trainer=None, iteration=10, scores=[1.0, 3.0, 5.0])

    plugin.on_done(trainer=None)
    mlflow.end_run()

    assert _metric_history(tmp_path, "eval/mean_score") == [(10, 3.0)]
    assert _metric_history(tmp_path, "eval/max_score") == [(10, 5.0)]
    assert _metric_history(tmp_path, "eval/min_score") == [(10, 1.0)]


def test_evaluation_hook_ignores_empty_scores(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.on_setup(trainer=None)
    plugin.on_eval(trainer=None, iteration=10, scores=[])
    plugin.on_done(trainer=None)
    mlflow.end_run()

    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    metric_keys = set(runs[0].data.metrics.keys())
    assert not any(k.startswith("eval/") for k in metric_keys)


def _artifact_dir_for_run(tmp_path):
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    return Path(runs[0].info.artifact_uri.replace("file://", ""))


def test_on_done_uploads_checkpoint_and_safetensors(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    st = tmp_path / "model.safetensors"
    st.write_bytes(b"fake-safetensors-bytes")

    plugin = _make_plugin(
        tmp_path,
        checkpoint_path=str(ckpt),
        safetensors_path=str(st),
    )
    plugin.on_setup(trainer=None)
    plugin.on_done(trainer=None)

    artifact_dir = _artifact_dir_for_run(tmp_path)
    assert (artifact_dir / "ckpt.pt").exists()
    assert (artifact_dir / "model.safetensors").exists()


def test_on_done_skips_missing_artifacts(tmp_path):
    plugin = _make_plugin(
        tmp_path,
        checkpoint_path=str(tmp_path / "does-not-exist.pt"),
        safetensors_path="",
    )
    plugin.on_setup(trainer=None)
    plugin.on_done(trainer=None)  # must not raise

    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    assert runs[0].info.status == "FINISHED"


def test_on_done_swallows_artifact_upload_errors(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"bytes")

    def _broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mlflow, "log_artifact", _broken)

    plugin = _make_plugin(tmp_path, checkpoint_path=str(ckpt))
    plugin.on_setup(trainer=None)
    plugin.on_done(trainer=None)  # must not raise despite broken log_artifact

    runs = mlflow.search_runs(experiment_names=["rlcade-test"], output_format="list")
    assert runs[0].info.status == "FINISHED"


def test_nonzero_rank_is_no_op(tmp_path, monkeypatch):
    """Non-rank-0 ranks must not start runs or log anything."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    forbidden = []
    for name in (
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_params",
        "log_metric",
        "log_artifact",
        "end_run",
    ):
        monkeypatch.setattr(mlflow, name, lambda *a, _n=name, **k: forbidden.append(_n))

    plugin = _make_plugin(tmp_path)
    plugin.on_setup(trainer=None)
    plugin.on_step_end(trainer=None, iteration=1, summary={"score": 1.0})
    plugin.on_eval(trainer=None, iteration=1, scores=[1.0, 2.0])
    plugin.on_done(trainer=None)

    assert forbidden == [], f"non-rank-0 leaked MLflow calls: {forbidden}"


def test_build_plugins_skips_mlflow_when_flag_unset(tmp_path):
    """If args.mlflow is None, rlcade.plugins.mlflow must NOT be imported."""
    sys.modules.pop("rlcade.plugins.mlflow", None)

    from rlcade.entrypoint import _build_plugins

    args = _make_args(
        mlflow=None,
        async_checkpoint=False,
        viztracer="",
        nsys=False,
        memory_profiler=False,
        safetensors_path="",
    )
    args.checkpoint_interval = 1
    args.tensorboard = None  # avoid TB SummaryWriter side-effects

    plugins = _build_plugins(args)
    assert "rlcade.plugins.mlflow" not in sys.modules
    assert not any(p.__class__.__name__ == "MLflowPlugin" for p in plugins)


def test_build_plugins_includes_mlflow_when_flag_set(tmp_path):
    from rlcade.entrypoint import _build_plugins

    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    args = _make_args(
        mlflow=tracking_uri,
        async_checkpoint=False,
        viztracer="",
        nsys=False,
        memory_profiler=False,
        safetensors_path="",
    )
    args.checkpoint_interval = 1
    args.tensorboard = None

    plugins = _build_plugins(args)
    mlflow_plugins = [p for p in plugins if p.__class__.__name__ == "MLflowPlugin"]
    assert len(mlflow_plugins) == 1
    assert mlflow_plugins[0]._tracking_uri == tracking_uri
    assert mlflow_plugins[0]._experiment == "rlcade"
    assert mlflow_plugins[0]._run_name is None
