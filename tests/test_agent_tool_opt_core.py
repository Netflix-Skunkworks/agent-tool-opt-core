"""Package-level smoke tests: the public API imports cleanly."""

import agent_tool_opt_core


def test_package_importable():
    assert agent_tool_opt_core


def test_optimizer_catalog_api():
    from agent_tool_opt_core.optimizers import build_optimizer, list_optimizers

    assert set(list_optimizers()) == {"llm", "draft", "pi", "gepa", "toolobserver"}
    assert build_optimizer("llm").id == "llm"


def test_api_imports():
    from agent_tool_opt_core.api import (
        Agent,
        Benchmark,
        Candidate,
        Metrics,
        Optimizer,
        RunResult,
        ToolSet,
        ToolTarget,
        ValidationResult,
        Validator,
        score,
    )

    assert all(
        x is not None
        for x in (
            Agent,
            Benchmark,
            Candidate,
            Metrics,
            Optimizer,
            RunResult,
            ToolSet,
            ToolTarget,
            ValidationResult,
            Validator,
            score,
        )
    )
