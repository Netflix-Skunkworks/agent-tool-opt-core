"""Offline DRAFT-inspired baseline on the ``api.Optimizer`` interface.

DRAFT (arXiv:2410.08197, "From Exploration to Mastery", ICLR 2025) refines tool
*documentation* through a per-tool trial-and-error loop of three phases — Explorer
(synthesize a query and CALL the live tool), Analyzer (compare the doc against the
tool's real output), Rewriter (revise the description) — iterated over a few
episodes with a diversity constraint and a BLEU/cosine-similarity termination.

This is deliberately an offline, non-faithful adaptation to the
``propose(tools, transcripts, validate)`` contract: we run a single Analyzer →
Rewriter pass over selected **baseline transcripts** (one failure per task plus up to
three unique-task successes) instead of the live
Explorer loop, and rewrite the whole tool file rather than each tool's description
field. Selected trajectories are lossless when they fit; on overflow, only long values
in structured trajectories are explicitly middle-elided. The Analyzer/Rewriter
criteria mirror the paper's — consistency with the tool's observed behavior,
comprehensiveness, conciseness. Full online exploration and iteration would need an
``Explorer`` capability (call a tool, observe its response, without seeing the reward)
— a future extension, intentionally not the metric.
"""

from __future__ import annotations

from pathlib import Path

from agent_tool_opt_core.api import Candidate, Optimizer, RunResult, ToolSet, Validator
from agent_tool_opt_core.llm_client import LLMClient, LiteLLMClient
from agent_tool_opt_core.optimizers._common import (
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    OptimizerLLMFailure,
    call_llm_with_retry,
    ensure_request_fits,
    fit_transcripts,
    model_context_window,
    request_token_count,
    single_editable_target,
    strip_code_fences,
)

_ANALYZE_SYS = (
    "You analyze an agent's runs to improve its tool documentation. Given the tool "
    "source and transcripts of the agent using the tools, suggest concrete changes "
    "to the tools' descriptions. Consider whether each description is consistent "
    "with how the tool actually behaves in the transcripts, whether it is "
    "comprehensive, and whether it is concise and free of irrelevant information."
)
_ANALYZE_USER = "## Tool source\n```\n{source}\n```\n\n## Transcripts\n{transcript}"

_REWRITE_SYS = (
    "You rewrite the tools to apply the suggested changes. Revise the descriptions "
    "to focus on what each tool does and how to use it correctly, omitting "
    "irrelevant details. Keep every tool's name and signature unchanged; do not add "
    "or remove tools. Output ONLY the complete modified file content."
)
_REWRITE_USER = (
    "## Current tool source\n```\n{source}\n```\n\n"
    "## Identified issues\n{analysis}\n\nReturn the full modified file content."
)


class DRAFTOptimizer(Optimizer):
    id = "draft"

    def __init__(
        self,
        model: str = DEFAULT_OPTIMIZER_MODEL,
        require_validation: bool = True,
        *,
        context_window_size: int | None = None,
        output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
        llm: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.require_validation = require_validation
        self.context_window_size = context_window_size
        self.output_reserve_tokens = output_reserve_tokens
        self.llm = llm or LiteLLMClient()

    def _complete(self, messages):
        ensure_request_fits(
            messages,
            self.model,
            output_reserve_tokens=self.output_reserve_tokens,
            context_window_size=self.context_window_size,
        )
        return call_llm_with_retry(
            lambda: self.llm.complete(model=self.model, messages=messages),
            model=self.model,
        )

    def propose(
        self,
        tools: ToolSet,
        run: RunResult,
        validate: Validator,
        scratch: Path,
        train_eval=None,  # blind optimizer: no train evaluator used
    ) -> Candidate:
        if not tools.allowlist:
            return Candidate({})
        target = single_editable_target(tools)
        source = tools.files.get(target, "")
        fixed_analysis_messages = [
            {"role": "system", "content": _ANALYZE_SYS},
            {
                "role": "user",
                "content": _ANALYZE_USER.format(source=source, transcript=""),
            },
        ]
        window = model_context_window(self.model, self.context_window_size)
        transcript_budget = (
            window
            - self.output_reserve_tokens
            - request_token_count(fixed_analysis_messages, self.model)
        )
        transcripts = fit_transcripts(
            run,
            token_budget=transcript_budget,
            model=self.model,
            optimizer=self.id,
        )
        analysis_messages = [
            {"role": "system", "content": _ANALYZE_SYS},
            {
                "role": "user",
                "content": _ANALYZE_USER.format(source=source, transcript=transcripts),
            },
        ]
        analysis = self._complete(analysis_messages).text

        rewrite_messages = [
            {
                "role": "system",
                "content": _REWRITE_SYS + "\n\n" + tools.language_rules,
            },
            {
                "role": "user",
                "content": _REWRITE_USER.format(source=source, analysis=analysis),
            },
        ]
        rewritten = self._complete(rewrite_messages).text
        candidate = Candidate({target: strip_code_fences(rewritten)})
        # single-shot: no retry loop, so a validation failure is terminal
        if self.require_validation and not validate.validate(candidate).ok:
            raise OptimizerLLMFailure(
                "draft optimizer: the rewritten file did not pass validation"
            )
        return candidate
