"""Metaflow passes the same public Pi sandbox configuration to each worker."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("metaflow")


def test_metaflow_forwards_sandbox_options_without_serializing_host_environment(
    tmp_path, monkeypatch
):
    source = Path(__file__).resolve().parents[1] / "harness" / "run_harness_metaflow.py"
    spec = importlib.util.spec_from_file_location("ato_sandbox_test_flow", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}

    def build(name, **kwargs):
        seen.update(name=name, **kwargs)
        return "optimizer"

    monkeypatch.setattr(module, "build_optimizer", build)
    state = SimpleNamespace(
        benchmark="tau2",
        optimizer="pi",
        optimizer_model="model",
        no_transcripts=False,
        no_validation=False,
        skip_optimize=False,
        pi_provider="openai",
        pi_sandbox="bubblewrap",
        pi_sandbox_env="OPENAI_API_KEY",
        artifact_only=True,
        methods="reward_shaping,generalization",
    )
    assert module.ToolOptimizationHarness._optimizer(state, tmp_path) == "optimizer"
    assert seen["sandbox"] == "bubblewrap"
    assert seen["sandbox_env"] == ("OPENAI_API_KEY",)
    assert seen["provider"] == "openai"
    assert "env" not in seen  # Resolve named secrets on the optimize worker.
