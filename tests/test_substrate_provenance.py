"""Provenance guard: benchmark- and strategy-specific priors are OPT-IN, never
baked into the shared optimizer substrate.

This encodes a design invariant: a deliberate prior about the target task
belongs in a NAMED, opt-in Method (``optimizers/catalog.py``) or arrives through
the ToolTarget's runtime channels (``ToolSet.language_rules`` /
``ToolSet.context``). It must not live in the shared substrate
(``BASE_OBJECTIVE`` and each optimizer's hardcoded prompt). The shared substrate
stays the same across benchmarks; only ToolTarget context and selected Methods
may vary.

This is NOT the anti-reward-hacking invariant. That one is structural and lives
elsewhere: blind optimizers receive a ``Validator`` and training transcripts;
search optimizers may also receive a training-only ``TrainEvaluator``. Neither
interface exposes held-out test evaluation. See
``test_driver_experiment.py::test_optimizer_only_receives_a_validator`` and
``::test_propose_rejects_test_split_run``, and
``test_train_evaluator.py::test_train_evaluator_scores_train_and_is_test_blind``.

Why provenance rather than a banned-word list: a vocabulary blacklist has false
negatives (a benchmark-specific prior that dodges the listed words passes) and
false positives (it blocks legitimate general wording). The property we actually
care about — and *can* enforce — is the channel a prior flows through, not which
words it uses.
"""

from __future__ import annotations

import pytest

from agent_tool_opt_core.api import RunResult, TaskRun, ToolSet, Validator
from agent_tool_opt_core.optimizers import (
    catalog,
    draft,
    gepa,
    llm,
    pi,
    toolobserver,
)
from agent_tool_opt_core.optimizers._common import BASE_OBJECTIVE
from agent_tool_opt_core.optimizers.llm import LLMOptimizer
from agent_tool_opt_core.testing import FakeLLMClient

# The shared substrate: prompts baked into the default path of every optimizer,
# independent of any benchmark or opt-in Method.
_SUBSTRATE = {
    "BASE_OBJECTIVE": BASE_OBJECTIVE,
    "pi._OBJECTIVE": pi._OBJECTIVE,
    "pi._COLD_PROMPT": pi._COLD_PROMPT,
    "pi._RETHINK_NUDGE": pi._RETHINK_NUDGE,
    "pi._TRANSCRIPTS_PROMPT": pi._TRANSCRIPTS_PROMPT,
    "pi._EVIDENCE_TRUST_RULES": pi._EVIDENCE_TRUST_RULES,
    "llm._SYSTEM": llm._SYSTEM,
    "gepa._REFLECT_SYSTEM": gepa._REFLECT_SYSTEM,
    "gepa._REFLECT_USER": gepa._REFLECT_USER,
    "gepa._REFLECT_USER_GENERALIZE": gepa._REFLECT_USER_GENERALIZE,
    "toolobserver._BATCH_SYSTEM": toolobserver._BATCH_SYSTEM,
    "toolobserver._BATCH_USER": toolobserver._BATCH_USER,
    "toolobserver._MERGE_SYSTEM": toolobserver._MERGE_SYSTEM,
    "toolobserver._MERGE_USER": toolobserver._MERGE_USER,
    "draft._ANALYZE_SYS": draft._ANALYZE_SYS,
    "draft._REWRITE_SYS": draft._REWRITE_SYS,
}

# The opt-in priors are exactly the catalog Methods' objective texts — derived
# from the live registry, not a hand-maintained list.
_METHOD_PRIORS = {
    name: m.objective for name, m in catalog._METHODS.items() if m.objective.strip()
}


def test_there_are_registered_method_priors():
    # Guards the guard: if the registry were empty, the substrate checks below
    # would vacuously pass.
    assert _METHOD_PRIORS, "expected at least one opt-in Method prior in the catalog"


@pytest.mark.parametrize("sub_name,text", list(_SUBSTRATE.items()))
def test_substrate_embeds_no_optin_prior(sub_name, text):
    """No shared-substrate prompt may embed a named Method's objective: those are
    opt-in priors and must stay in the catalog, out of the default path."""
    for prior_name, prior_text in _METHOD_PRIORS.items():
        assert prior_text.strip() not in text, (
            f"{sub_name} embeds the opt-in prior '{prior_name}' — keep it in its "
            f"catalog Method, not in the shared substrate."
        )


# ---- assembly-level provenance (llm is the representative method-aware path) ----


def _llm_system(mocker, tmp_path, *, language_rules="RULES", method="", run=None):
    """Assemble and capture the LLM optimizer's system prompt."""
    fake = FakeLLMClient(["def t():\n    return 1\n"])
    tools = ToolSet({"tools.py": "def t(): pass\n"}, ("tools.py",), language_rules)
    run = run or RunResult("b", "g", (TaskRun("t1", 0.0, trajectory="tried x"),))
    LLMOptimizer(method_addendum=method, llm=fake).propose(
        tools, run, Validator("py", ("tools.py",)), tmp_path
    )
    return fake.calls[0]["messages"][0]["content"]


def test_system_substrate_is_invariant_across_benchmarks(mocker, tmp_path):
    """The assembled SYSTEM substrate does not depend on the benchmark or its
    transcripts — swapping the run (name + trajectories) leaves it identical."""
    run_a = RunResult("airline", "g", (TaskRun("a", 0.0, trajectory="cancel booking"),))
    run_b = RunResult("term", "g", (TaskRun("b", 0.0, trajectory="run bash pytest"),))
    sys_a = _llm_system(mocker, tmp_path, language_rules="RULES", run=run_a)
    sys_b = _llm_system(mocker, tmp_path, language_rules="RULES", run=run_b)
    assert sys_a == sys_b  # benchmark content never reaches the substrate


def test_language_rules_is_the_only_toolkind_channel(mocker, tmp_path):
    """Tool-kind specifics reach the prompt ONLY through ToolSet.language_rules:
    swapping that string changes the system prompt by exactly that text."""
    sys_a = _llm_system(mocker, tmp_path, language_rules="RULES_ALPHA")
    sys_b = _llm_system(mocker, tmp_path, language_rules="RULES_BRAVO")
    assert "RULES_ALPHA" in sys_a and "RULES_BRAVO" in sys_b
    assert sys_a.replace("RULES_ALPHA", "") == sys_b.replace("RULES_BRAVO", "")


def test_method_prior_is_opt_in_and_purely_additive(mocker, tmp_path):
    """A Method's objective reaches the prompt iff it is selected: the default
    (no-method) substrate carries no opt-in prior, and selecting one only appends
    its text."""
    plain = _llm_system(mocker, tmp_path, language_rules="R")
    for prior in _METHOD_PRIORS.values():
        assert prior.strip() not in plain  # default path is prior-free
    withm = _llm_system(mocker, tmp_path, language_rules="R", method="INJECTED_PRIOR")
    assert withm == plain + "\n\n" + "INJECTED_PRIOR"  # opt-in, purely additive
