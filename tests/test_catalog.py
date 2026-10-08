"""Tests for the new-optimizer catalog / build_optimizer factory."""

from __future__ import annotations

import pytest

from agent_tool_opt_core.optimizers.catalog import (
    build_optimizer,
    list_methods,
    list_optimizers,
)
from agent_tool_opt_core.optimizers.draft import DRAFTOptimizer
from agent_tool_opt_core.optimizers.llm import LLMOptimizer
from agent_tool_opt_core.optimizers.reward_shaping import REWARD_SHAPING_OBJECTIVE


def test_list_optimizers():
    assert set(list_optimizers()) == {"llm", "draft", "pi", "gepa", "toolobserver"}


def test_build_llm_and_draft():
    assert isinstance(build_optimizer("llm", model="m"), LLMOptimizer)
    assert isinstance(build_optimizer("draft"), DRAFTOptimizer)


def test_build_unknown_raises():
    with pytest.raises(KeyError, match="Unknown optimizer"):
        build_optimizer("nope")


def test_build_pi_with_reward_shaping(mocker):
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.shutil.which",
        return_value="/usr/local/bin/pi",
    )
    opt = build_optimizer("pi", reward_shaping=True)
    assert opt.id == "pi"
    assert opt.method == REWARD_SHAPING_OBJECTIVE  # rich objective
    assert opt.context_builder is not None  # run-derived context engine wired


def test_build_pi_plain(mocker):
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.shutil.which",
        return_value="/usr/local/bin/pi",
    )
    opt = build_optimizer("pi")
    assert opt.method == ""
    assert opt.context_builder is None
    # defaults: transcripts on, validation on (scope is the ToolTarget's concern)
    assert opt.use_transcripts and opt.require_validation


# ---- composable methods (ablation axes) ------------------------------------


def _pi(mocker):
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.shutil.which",
        return_value="/usr/local/bin/pi",
    )


def test_list_methods():
    assert set(list_methods()) == {"reward_shaping", "generalization"}


def test_compose_two_methods(mocker):
    _pi(mocker)
    opt = build_optimizer("pi", methods=["reward_shaping", "generalization"])
    assert "on disk as evidence" in opt.method  # reward_shaping objective
    assert "held-out tasks" in opt.method  # generalization objective
    assert opt.context_builder is not None  # reward_shaping contributes context


def test_generalization_only_has_no_context(mocker):
    _pi(mocker)
    opt = build_optimizer("pi", methods=["generalization"])
    assert "held-out tasks" in opt.method
    assert opt.context_builder is None  # generalization is prompt-only


def test_unknown_method_raises(mocker):
    _pi(mocker)
    with pytest.raises(KeyError, match="Unknown method"):
        build_optimizer("pi", methods=["bogus"])


def test_methods_compose_for_llm():
    opt = build_optimizer("llm", methods=["reward_shaping", "generalization"])
    assert "on disk as evidence" in opt.method
    assert "held-out tasks" in opt.method
    assert opt.context_builder is not None  # reward_shaping context, shared with pi


def test_methods_rejected_for_non_method_aware():
    with pytest.raises(KeyError, match="only supported for"):
        build_optimizer("draft", methods=["reward_shaping"])


def test_ablation_kwargs_passthrough(mocker):
    _pi(mocker)
    opt = build_optimizer("pi", use_transcripts=False, require_validation=False)
    assert opt.use_transcripts is False
    assert opt.require_validation is False
    # llm gets the same knobs
    lopt = build_optimizer("llm", use_transcripts=False, require_validation=False)
    assert lopt.use_transcripts is False and lopt.require_validation is False


def test_gepa_accepts_generalization_method():
    from agent_tool_opt_core.optimizers.gepa import (
        _REFLECT_USER,
        _REFLECT_USER_GENERALIZE,
    )

    plain = build_optimizer("gepa")
    assert plain.generalize is False and plain.reflect_user is _REFLECT_USER
    gen = build_optimizer("gepa", methods=["generalization"])
    assert gen.generalize is True  # method -> transfer-first reflection prompt
    assert gen.reflect_user is _REFLECT_USER_GENERALIZE


def test_gepa_rejects_context_materializing_methods():
    # reward_shaping needs workspace files gepa never materializes.
    with pytest.raises(KeyError, match="gepa supports only"):
        build_optimizer("gepa", methods=["reward_shaping"])
    with pytest.raises(KeyError, match="gepa supports only"):
        build_optimizer("gepa", reward_shaping=True)
