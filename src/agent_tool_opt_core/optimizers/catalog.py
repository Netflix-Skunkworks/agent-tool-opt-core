"""Catalog of optimizers + composable *methods* (the ablation surface).

An optimizer is selected by id; ablations are set by composing orthogonal knobs
on ``build_optimizer`` — no new class per combination:

  - ``methods=(...)``    — insight bundles, each an objective addendum + optional
                           run-derived read-only context. Composable (axes 4/5):
                           ``"reward_shaping"``, ``"generalization"``.
  - ``use_transcripts``  — feed the baseline transcripts (prompt + method context)
                           vs. optimize blind from the source (axis 1).
  - ``require_validation`` — gate candidates on the language Validator (axis 3).

These apply to both agent (``pi``) and one-shot (``llm``) techniques. Edit
*scope* (axis 2: full vs docstrings-only) is NOT a knob here — it is owned by
the ToolTarget (a descriptions-only ToolTarget exposes only the docstrings as
the editable surface), so it ablates uniformly across every optimizer.

Example ablation cells::

    build_optimizer("pi")                                  # base
    build_optimizer("pi", methods=["reward_shaping"])      # + reward shaping
    build_optimizer("llm", methods=["reward_shaping", "generalization"])
    build_optimizer("pi", use_transcripts=False)           # blind
    build_optimizer("llm", require_validation=False)       # no validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent_tool_opt_core.api import Optimizer, RunResult
from agent_tool_opt_core.optimizers.draft import DRAFTOptimizer
from agent_tool_opt_core.optimizers.gepa import GEPAOptimizer
from agent_tool_opt_core.optimizers.llm import LLMOptimizer
from agent_tool_opt_core.optimizers.toolobserver import ToolObserverOptimizer
from agent_tool_opt_core.optimizers.reward_shaping import (
    REWARD_SHAPING_OBJECTIVE,
    build_reward_shaping_context,
)

# A generalization-friendly insight set — a sibling method to reward shaping,
# focused on edits that transfer to held-out tasks rather than the observed ones.
GENERALIZATION_OBJECTIVE = """\
The tools are scored on held-out tasks, not the ones in these transcripts. Prefer
edits that address the general defect a transcript reveals over edits that only fit
the specific observed case."""


@dataclass(frozen=True)
class Method:
    """A composable optimization insight: a prompt addendum + an optional
    run-derived read-only context builder. An ablation includes it or not."""

    name: str
    objective: str = ""
    context_builder: Callable[[RunResult], dict[str, str]] | None = None


_METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        Method(
            "reward_shaping", REWARD_SHAPING_OBJECTIVE, build_reward_shaping_context
        ),
        Method("generalization", GENERALIZATION_OBJECTIVE, None),
    )
}


def list_methods() -> list[str]:
    return sorted(_METHODS)


def _compose_methods(
    names: list[str],
) -> tuple[str, Callable[[RunResult], dict[str, str]] | None]:
    """Combine method addenda (concatenated) + context builders (merged)."""
    chosen = []
    for n in names:
        if n not in _METHODS:
            raise KeyError(f"Unknown method '{n}'. Available: {list_methods()}")
        chosen.append(_METHODS[n])
    objective = "\n\n".join(m.objective for m in chosen if m.objective)
    builders = [m.context_builder for m in chosen if m.context_builder is not None]
    if not builders:
        return objective, None
    if len(builders) == 1:
        return objective, builders[0]

    def merged(run: RunResult) -> dict[str, str]:
        out: dict[str, str] = {}
        for b in builders:
            out.update(b(run))
        return out

    return objective, merged


# Optimizers that compose the full method surface (objective addendum + context).
_METHOD_AWARE = ("pi", "llm")
# ``gepa`` is a search optimizer (wants_train_eval) with a bespoke loop; it honors
# only the prompt-only ``generalization`` method (by swapping its reflection prompt),
# not the context-materializing ``reward_shaping``.
_GEPA_METHODS = ("generalization",)
# Bespoke-loop optimizers (no method composition).
_BUILDERS = {
    "draft": DRAFTOptimizer,
    "toolobserver": ToolObserverOptimizer,
    "gepa": GEPAOptimizer,
}


def list_optimizers() -> list[str]:
    return sorted([*_BUILDERS, *_METHOD_AWARE])


def build_optimizer(
    name: str,
    *,
    methods: list[str] | tuple[str, ...] = (),
    reward_shaping: bool = False,
    **kwargs,
) -> Optimizer:
    """Construct an ``Optimizer`` by id, composing ablation methods.

    ``methods`` (and the ``reward_shaping=True`` shorthand) compose insight bundles
    into the ``pi``/``llm`` optimizer's objective + context. ``gepa`` accepts only
    ``methods=['generalization']`` — it has no method-composition surface, so it maps
    that one method to swapping its reflection prompt. ``"pi"`` is imported lazily (it
    needs the ``pi`` binary). Other ablation axes (``use_transcripts``,
    ``require_validation``) pass straight through as kwargs.
    """
    names = list(methods)
    if reward_shaping and "reward_shaping" not in names:
        names.append("reward_shaping")  # shorthand / back-compat

    if name in _METHOD_AWARE:
        objective, context_builder = _compose_methods(names)
        if objective:
            kwargs.setdefault("method_addendum", objective)
        if context_builder is not None:
            kwargs.setdefault("context_builder", context_builder)
        if name == "llm":
            return LLMOptimizer(**kwargs)
        from agent_tool_opt_core.optimizers.pi import PiOptimizer  # needs the binary

        return PiOptimizer(**kwargs)

    if name == "gepa":
        unsupported = [n for n in names if n not in _GEPA_METHODS]
        if unsupported:
            raise KeyError(
                f"gepa supports only {list(_GEPA_METHODS)} (it swaps its reflection "
                f"prompt); got {unsupported}. reward_shaping needs the workspace "
                f"materialization only {list(_METHOD_AWARE)} provide."
            )
        if "generalization" in names:
            kwargs.setdefault("generalize", True)
        return GEPAOptimizer(**kwargs)

    if names:
        raise KeyError(
            f"methods are only supported for {(*_METHOD_AWARE, 'gepa')}, not '{name}'"
        )
    if name not in _BUILDERS:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {list_optimizers()}")
    return _BUILDERS[name](**kwargs)
