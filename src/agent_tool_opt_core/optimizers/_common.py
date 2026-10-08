"""Shared helpers for the ``api.Optimizer`` implementations.

Used by llm / draft / pi (the prompt + objective helpers) and by gepa /
toolobserver (``render_trajectory``, ``BASE_OBJECTIVE``)."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from importlib import import_module
from typing import Callable, TypeVar

from agent_tool_opt_core.api import RunResult, ToolSet
from agent_tool_opt_core.costs import record_call_failure, record_completion

logger = logging.getLogger(__name__)

# Default LLM used by the optimizers (llm / draft / gepa / toolobserver / pi) to
# rewrite tools. Provider-qualified names keep routing explicit. Callers can
# override this per run or set ``AGENT_TOOL_OPT_MODEL`` at the composition root.
DEFAULT_OPTIMIZER_MODEL = os.environ.get(
    "AGENT_TOOL_OPT_MODEL", "anthropic/claude-sonnet-4-5"
)

DEFAULT_OUTPUT_RESERVE_TOKENS = 32_768

# LiteLLM may use an approximation for Claude models rather than Anthropic's
# server-side tokenizer. Keep conservative headroom; OpenAI models with an exact
# tiktoken encoding do not need this adjustment.
_CLAUDE_TOKEN_SAFETY_FACTOR = 1.6

# The objective every optimizer shares: the inputs, the goal, a benchmark-agnostic
# principle of what makes a toolset effective, and the rules of the game (interface
# + anti-cheat). It deliberately says nothing about the target domain or about the
# specific edit to make — those are priors about *this* task, which the optimizer
# must infer from the evidence (guarded by tests/test_substrate_provenance.py).
# A general principle of tool effectiveness is not such a prior: it holds on any
# benchmark.
BASE_OBJECTIVE = """\
You are given an agent's tools (source you may edit) and transcripts of the agent
attempting tasks with them, each labelled with its reward. Edit the tools so the
agent succeeds on more tasks.

Read the transcripts for places where the tools — not the agent's judgment — got in
the way. Look for: the agent picking the wrong tool or wrong arguments because the
docs don't make a tool's purpose or usage clear; results it can't act on (unclear,
incomplete, or silent) that force it to guess or retry; and repeated calls or
workarounds for something the tools should let it do in one step. Effective tools
make the right action easy to choose, behave as their docs describe, and hand back
what the agent needs to take the next step. Edit the tools to remove those obstacles,
and read the successful transcripts too, so a change doesn't break what already works.

Constraints: keep every tool's name and signature unchanged and the source valid;
do not encode task identifiers, specific inputs, or expected answers into the tools."""


class OptimizerLLMFailure(Exception):
    """The optimizer's own LLM call failed in a way that should score the
    candidate as a loss (reward 0) rather than crash the whole run or silently
    fall back to "no edit" (which would tie with baseline instead of losing).

    Raised for deterministic candidate failures such as a context-window
    overflow or validation exhaustion. Transient provider failures use
    ``OptimizerInfrastructureFailure`` so orchestration can retry the complete
    proposal without turning an outage into reward zero.
    """


class OptimizerInfrastructureFailure(Exception):
    """A transient optimizer dependency remained unavailable after call retries."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def classify_llm_error(exc: BaseException) -> str:
    """Classify an LLM call failure: "transient" (rate limit / timeout / 5xx /
    overload — retry), "context" (the prompt overflowed the model's context
    window — the optimizer's prompt is too big, not a bug), or "other" (anything
    else — a genuine bug, let it crash immediately rather than retry or mask it).

    Duck-typed on common OpenAI-compatible exception attributes so this module
    does not depend on provider-specific exception classes.
    """
    status = getattr(exc, "status_code", None)
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()

    # Some gateways wrap a context-window rejection in a 5xx response. The
    # message is more specific than the transport status: context exhaustion is
    # a deterministic candidate failure and must not be retried or excluded as
    # an infrastructure outage.
    if (
        "context_length_exceeded" in msg
        or "context window" in msg
        or "maximum context length" in msg
        or "prompt is too long" in msg
        or "too many tokens" in msg
    ):
        return "context"
    if (
        status == 429
        or "ratelimit" in cls_name
        or "rate_limit" in msg
        or "rate limit" in msg
    ):
        return "transient"
    if (
        status == 408
        or (isinstance(status, int) and 500 <= status < 600)
        or "timeout" in cls_name
        or "connection" in cls_name
        or "overload" in msg
        or "529" in msg
    ):
        return "transient"
    return "other"


_T = TypeVar("_T")


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Read OpenAI-compatible ``Retry-After`` metadata, when supplied."""
    value = getattr(exc, "retry_after", None)
    if value is None:
        headers = getattr(exc, "headers", None)
        response = getattr(exc, "response", None)
        if headers is None and response is not None:
            headers = getattr(response, "headers", None)
        if headers is not None:
            value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(value))
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def call_llm_with_retry(
    fn: Callable[[], _T],
    *,
    model: str | None = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
) -> _T:
    """Call an optimizer's LLM completion ``fn``, retrying transient failures
    with capped exponential backoff.

    - "transient" (rate limit / timeout / 5xx / overload): retry up to
      ``max_retries`` attempts; on exhaustion, raise
      ``OptimizerInfrastructureFailure`` so the orchestration layer can retry
      the whole proposal.
    - "context": raise ``OptimizerLLMFailure`` immediately, no retry.
    - "other": re-raise immediately — a genuine bug, cheap to notice and rerun.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if base_delay < 0 or max_delay < 0 or jitter < 0:
        raise ValueError("retry delays and jitter must not be negative")
    delay = base_delay
    last_exc: BaseException | None = None
    retry_after: float | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = fn()
        except Exception as e:
            record_call_failure(e)
            kind = classify_llm_error(e)
            if kind == "context":
                raise OptimizerLLMFailure(
                    f"optimizer LLM call exceeded context window: {e}"
                ) from e
            if kind == "other":
                raise
            last_exc = e
            retry_after = _retry_after_seconds(e)
            if attempt == max_retries:
                break
            wait = max(delay, retry_after or 0.0)
            wait += random.uniform(0.0, wait * jitter)
            logger.warning(
                "optimizer LLM call hit a transient error (attempt %d/%d), "
                "retrying in %.0fs: %s",
                attempt, max_retries, wait, e,
            )  # fmt: skip
            time.sleep(wait)
            delay = min(delay * 2, max_delay)
        else:
            record_completion(response, model)
            return response
    raise OptimizerInfrastructureFailure(
        f"optimizer LLM call failed after {max_retries} attempts (transient): {last_exc}",
        retry_after=retry_after,
    ) from last_exc


# Per-value character caps tried in turn by the opt-in fitter, largest first.
_ABBREV_LADDER = (700, 500, 350, 240, 160, 110, 70, 40, 28, 20)

_ELIDED = "…[{n} chars elided]…"
_ABBREV_NOTICE = (
    "\n…({n} long values in this transcript were middle-elided; "
    "every field is still present)"
)


def _dump(traj: object, indent: int | None = 2) -> str:
    """A raw trajectory as text: a string as-is, anything else JSON-serialized.

    ``indent=None`` drops the pretty-printing whitespace, which on a deep
    trajectory is a third of the bytes and none of the evidence.
    """
    if isinstance(traj, str):
        return traj
    try:
        return json.dumps(
            traj,
            indent=indent,
            separators=None if indent else (",", ": "),
            default=str,
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return str(traj)


def _abbreviate(s: str, cap: int) -> str:
    """Keep ``cap`` source characters from ``s`` plus an explicit gap marker.

    Both ends, not a prefix: the evidence an optimizer is looking for (the error
    a call ended on) tends to sit at the end of a long value, which is exactly
    what prefix truncation throws away.
    """
    marker = _ELIDED.format(n=len(s) - cap)
    if len(s) <= cap + len(marker):
        return s  # abbreviating would not actually save anything
    tail = cap // 3
    return s[: cap - tail] + marker + s[len(s) - tail :]


def _shrink(obj: object, cap: int, shrunk: list[int]) -> object:
    """``obj`` with every string leaf longer than ``cap`` abbreviated.

    Keys, containers and non-string scalars are untouched, so the shape of the
    trajectory — every tool call, its arguments, its verdict — survives at any
    cap; only bulk (result bodies, long argument values) gives way. Which leaves
    are bulky is decided by length alone, so this stays domain-agnostic: it needs
    no idea of what the benchmark calls its fields.
    """
    if isinstance(obj, str):
        if len(obj) <= cap:
            return obj
        out = _abbreviate(obj, cap)
        if out != obj:
            shrunk[0] += 1
        return out
    if isinstance(obj, dict):
        return {k: _shrink(v, cap, shrunk) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_shrink(v, cap, shrunk) for v in obj]
    return obj


def render_trajectory(traj: object) -> str:
    """Losslessly serialize a benchmark's raw transcript for an optimizer.

    The adapter stores whatever the benchmark emits, verbatim (see
    ``api.TaskRun.trajectory``); this is the only place that turns it into text,
    and it does so uniformly — a string is used as-is, any other structure is
    JSON pretty-printed. No benchmark-specific field knowledge, so the optimizer
    cannot tell which benchmark it is looking at from our processing. Lossy
    policy belongs in ``fit_trajectory`` / ``fit_transcripts`` and must be chosen
    explicitly by an optimizer.
    """
    if traj is None:
        return ""
    return _dump(traj)


@lru_cache(maxsize=1)
def _litellm():
    """Load LiteLLM against its bundled model map, without a network refresh."""
    key = "LITELLM_LOCAL_MODEL_COST_MAP"
    previous = os.environ.get(key)
    os.environ[key] = "true"
    try:
        return import_module("litellm")
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def text_token_count(text: str, model: str) -> int:
    """Estimate tokens with model-specific tokenization and safety headroom."""
    count = int(_litellm().token_counter(model=model, text=text))
    if "claude" in model.lower():
        return math.ceil(count * _CLAUDE_TOKEN_SAFETY_FACTOR)
    return count


def model_context_window(model: str, override: int | None = None) -> int:
    """Resolve a model's input window from LiteLLM or an explicit override."""
    if override is not None:
        if override <= 0:
            raise ValueError("context_window_size must be positive")
        return override
    try:
        size = _litellm().get_model_info(model).get("max_input_tokens")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("missing max_input_tokens")
        return size
    except Exception as exc:
        raise ValueError(
            f"unknown context window for optimizer model {model!r}; pass "
            "context_window_size explicitly"
        ) from exc


def request_token_count(messages: list[dict[str, str]], model: str) -> int:
    """Estimate the chat-message tokens, padding known approximate tokenizers."""
    count = int(_litellm().token_counter(model=model, messages=messages))
    if "claude" in model.lower():
        return math.ceil(count * _CLAUDE_TOKEN_SAFETY_FACTOR)
    return count


def ensure_request_fits(
    messages: list[dict[str, str]],
    model: str,
    *,
    output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
    context_window_size: int | None = None,
) -> int:
    """Preflight a complete request or raise ``OptimizerLLMFailure``.

    Returns the safety-adjusted model-specific input-token estimate.
    """
    if output_reserve_tokens < 0:
        raise ValueError("output_reserve_tokens must be non-negative")
    window = model_context_window(model, context_window_size)
    used = request_token_count(messages, model)
    if used + output_reserve_tokens > window:
        raise OptimizerLLMFailure(
            f"optimizer request uses {used} input tokens plus "
            f"{output_reserve_tokens} reserved output tokens, exceeding "
            f"{model!r}'s {window}-token context window"
        )
    return used


def _fitted_render(traj: object, cap: int, indent: int | None) -> str:
    # A top-level string has no structure whose boundaries we can preserve.
    if isinstance(traj, str):
        return traj
    shrunk = [0]
    text = _dump(_shrink(traj, cap, shrunk), indent)
    notice = _ABBREV_NOTICE.format(n=shrunk[0]) if shrunk[0] else ""
    return text + notice


def _log_fitting(
    *,
    optimizer: str,
    task_id: object,
    model: str,
    original: str,
    fitted: str,
    strategy: str,
) -> None:
    logger.warning(
        "optimizer=%s task=%s model=%s fitted transcript: %d -> %d tokens (%s)",
        optimizer,
        task_id,
        model,
        text_token_count(original, model),
        text_token_count(fitted, model),
        strategy,
    )


def _fit_strategy(text: str, cap: int, indent: int | None) -> str:
    if "long values in this transcript were middle-elided" in text:
        return (
            f"structured-middle-elision cap={cap} "
            f"format={'pretty' if indent else 'compact'}"
        )
    return "compact-json"


def fit_trajectory(
    traj: object,
    *,
    token_budget: int,
    model: str = DEFAULT_OPTIMIZER_MODEL,
    optimizer: str = "unknown",
    task_id: object = "unknown",
) -> str:
    """Explicit structural middle-elision policy; never prefix-truncates.

    Opaque strings cannot be compressed without discarding an unknown part of
    their evidence, so an oversized opaque trajectory fails visibly.
    """
    if token_budget <= 0:
        raise OptimizerLLMFailure("no request budget remains for this trajectory")
    full = render_trajectory(traj)
    if text_token_count(full, model) <= token_budget:
        return full
    if isinstance(traj, str):
        raise OptimizerLLMFailure(
            "opaque trajectory does not fit its token budget and cannot be "
            "structurally compressed"
        )
    for cap in _ABBREV_LADDER:
        for indent in (2, None):
            text = _fitted_render(traj, cap, indent)
            if text_token_count(text, model) <= token_budget:
                _log_fitting(
                    optimizer=optimizer,
                    task_id=task_id,
                    model=model,
                    original=full,
                    fitted=text,
                    strategy=_fit_strategy(text, cap, indent),
                )
                return text
    raise OptimizerLLMFailure(
        "trajectory structure does not fit its token budget even after "
        "middle-eliding every long string value"
    )


def _selected_runs(run: RunResult, max_passing: int) -> list:
    # A RunResult may contain multiple trials for the same task. One-shot
    # optimizers need task coverage, not five copies of the same task consuming
    # the context window. Keep the first failure per task plus a bounded first
    # success per task; the adapter's stable run order makes this reproducible.
    failing = []
    seen_failing = set()
    passing = []
    seen_passing = set()
    for r in run.runs:
        if r.reward < 1.0:
            if r.task_id not in seen_failing:
                failing.append(r)
                seen_failing.add(r.task_id)
        elif r.task_id not in seen_passing and len(passing) < max_passing:
            passing.append(r)
            seen_passing.add(r.task_id)
    return failing + passing


def _transcript_summary(runs: list, render: Callable[[object], str]) -> str:
    lines: list[str] = []
    for r in runs:
        outcome = "PASS" if r.reward >= 1.0 else "FAIL"
        lines.append(
            f"### task {r.task_id} [{outcome}] (reward={r.reward})\n"
            f"{render(r.trajectory)}"
        )
    return "\n".join(lines) if lines else "(no transcripts)"


def summarize_transcripts(
    run: RunResult,
    max_passing: int = 3,
) -> str:
    """Render one failure per task, then a capped unique-task success sample."""
    return _transcript_summary(_selected_runs(run, max_passing), render_trajectory)


def fit_transcripts(
    run: RunResult,
    *,
    token_budget: int,
    model: str = DEFAULT_OPTIMIZER_MODEL,
    max_passing: int = 3,
    optimizer: str = "unknown",
) -> str:
    """Fit the selected transcript set to one total budget, structurally.

    A single cap is applied across the set so evidence is compressed uniformly.
    If even the compact call/result skeleton cannot fit, fail rather than return
    a biased prefix.
    """
    if token_budget <= 0:
        raise OptimizerLLMFailure("no request budget remains for transcripts")
    runs = _selected_runs(run, max_passing)
    full = _transcript_summary(runs, render_trajectory)
    if text_token_count(full, model) <= token_budget:
        return full
    for cap in _ABBREV_LADDER:
        for indent in (2, None):
            fitted = _transcript_summary(
                runs, lambda traj: _fitted_render(traj, cap, indent)
            )
            if text_token_count(fitted, model) <= token_budget:
                for r in runs:
                    original_body = render_trajectory(r.trajectory)
                    fitted_body = _fitted_render(r.trajectory, cap, indent)
                    if fitted_body != original_body:
                        _log_fitting(
                            optimizer=optimizer,
                            task_id=r.task_id,
                            model=model,
                            original=original_body,
                            fitted=fitted_body,
                            strategy=_fit_strategy(fitted_body, cap, indent),
                        )
                return fitted
    raise OptimizerLLMFailure(
        "selected transcript skeletons do not fit the total request budget"
    )


def transcript_workspace_files(run: RunResult) -> dict[str, str]:
    """Full per-task evidence plus a short stable index for coding agents."""
    files: dict[str, str] = {}
    index = ["# Baseline transcripts\n"]
    for i, r in enumerate(run.runs):
        outcome = "passed" if r.reward >= 1.0 else "failed"
        suffix = "txt" if isinstance(r.trajectory, str) else "json"
        rel = f"baseline_transcripts/{i:04d}.{suffix}"
        files[rel] = render_trajectory(r.trajectory)
        index.append(f"- `{rel}`: task `{r.task_id}`; {outcome}; reward={r.reward}\n")
    files["BASELINE_TRANSCRIPTS_INDEX.md"] = "".join(index)
    return files


_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_UNCLOSED_FENCE_RE = re.compile(r"```[^\n`]*\n(.*)\Z", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Extract the contents of a fenced code block, wherever it appears in
    ``text``; fall back to the whole trimmed text if there is no fence.

    The fence needn't start the response — models often open with a preamble
    like "Here's the updated file:". Of multiple blocks, the largest is assumed
    to be the file (rationale snippets are much shorter). A response truncated
    at ``max_tokens`` leaves the fence unclosed; that case is matched separately
    so the ``` marker doesn't leak in and guarantee an unparseable candidate.
    """
    matches = _FENCE_RE.findall(text)
    if matches:
        t = max(matches, key=len)
    else:
        unclosed = _UNCLOSED_FENCE_RE.search(text)
        t = unclosed.group(1) if unclosed else text.strip()
    t = t.strip()
    if t and not t.endswith("\n"):
        t += "\n"
    return t


def single_editable_target(tools: ToolSet) -> str:
    """The one editable relpath; raises if the target isn't exactly one file.

    ``llm``/``draft`` rewrite a single file; multi-file targets (opencode,
    ToolBench descriptions) need ``pi`` or a benchmark-specific strategy.
    """
    if len(tools.allowlist) != 1:
        raise ValueError(
            f"this optimizer rewrites a single file, but the target has "
            f"{len(tools.allowlist)} editable files {list(tools.allowlist)} — "
            f"use the 'pi' optimizer for multi-file targets."
        )
    return tools.allowlist[0]
