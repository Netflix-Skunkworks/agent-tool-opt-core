"""TauBench Verified adapter; upstream ``tau2`` is installed separately.

Description-only mode checks candidate source against the baseline AST before
execution. The opt-in full-code mode preserves the @is_tool contracts but can
change implementations. The upstream checkout is never written to disk. Runs
sharing a process must be sequential: tau2's environment module holds the
active tools class globally.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
    ValidationResult,
    Validator,
)
from agent_tool_opt_core.costs import total, usd

_DOMAINS = {
    "airline": "AirlineTools",
    "retail": "RetailTools",
    "telecom": "TelecomTools",
}
_DESCRIPTION_RULES = (
    "Edit only docstrings on @is_tool methods in tools.py. Keep imports, "
    "classes, method signatures, decorators, and executable code unchanged. "
    "Preserve the meaning and required arguments of every tool."
)
_CODE_RULES = (
    "Edit tools.py while preserving the tools class, every @is_tool method name, "
    "decorator, signature, and return annotation. Keep the module importable."
)


def _is_tool(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            getattr(target, "id", None) == "is_tool"
            or getattr(target, "attr", None) == "is_tool"
        ):
            return True
    return False


def _tool_contracts(source: str, class_name: str) -> dict[str, str]:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one {class_name} class")
    contracts = {}
    for node in classes[0].body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_tool(node):
            if node.name in contracts:
                raise ValueError(f"duplicate @is_tool method: {node.name}")
            contracts[node.name] = ast.dump(
                ast.Tuple(
                    elts=[
                        node.args,
                        node.returns or ast.Constant(None),
                        ast.List(elts=node.decorator_list, ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                ),
                include_attributes=False,
            )
    if not contracts:
        raise ValueError("tools class has no @is_tool methods")
    return contracts


def _without_tool_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_tool(node) and node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                value.value = ""
    return ast.dump(tree, include_attributes=False)


@dataclass(frozen=True)
class Tau2DescriptionValidator(Validator):
    baseline_source: str = ""

    def validate(self, candidate: Candidate) -> ValidationResult:
        basic = super().validate(candidate)
        if not basic.ok:
            return basic
        source = candidate.files.get("tools.py")
        if source is None:
            return ValidationResult(True, "no change")
        try:
            if _without_tool_docstrings(source) != _without_tool_docstrings(
                self.baseline_source
            ):
                return ValidationResult(False, "only @is_tool docstrings may change")
            compile(source, "tools.py", "exec")
        except (SyntaxError, ValueError) as exc:
            return ValidationResult(False, f"invalid Python source: {exc}")
        return ValidationResult(True, "docstrings-only edit")


@dataclass(frozen=True)
class Tau2CodeValidator(Validator):
    baseline_source: str = ""
    class_name: str = ""

    def validate(self, candidate: Candidate) -> ValidationResult:
        basic = super().validate(candidate)
        if not basic.ok:
            return basic
        source = candidate.files.get("tools.py")
        if source is None:
            return ValidationResult(True, "no change")
        try:
            if _tool_contracts(source, self.class_name) != _tool_contracts(
                self.baseline_source, self.class_name
            ):
                return ValidationResult(False, "@is_tool contracts changed")
            compile(source, "tools.py", "exec")
        except (SyntaxError, ValueError) as exc:
            return ValidationResult(False, f"invalid Python source: {exc}")
        return ValidationResult(True, "tool contracts preserved")


class Tau2ToolTarget(ToolTarget):
    kind = "py"
    language_rules = _DESCRIPTION_RULES

    def __init__(
        self, domain: str = "airline", *, descriptions_only: bool = True
    ) -> None:
        if domain not in _DOMAINS:
            raise ValueError(f"unsupported tau2 domain: {domain}")
        self.domain = domain
        self.descriptions_only = descriptions_only
        self.language_rules = _DESCRIPTION_RULES if descriptions_only else _CODE_RULES
        self._class_name = _DOMAINS[domain]
        self._module_name = f"tau2.domains.{domain}.tools"
        self._environment = importlib.import_module(
            f"tau2.domains.{domain}.environment"
        )
        module = importlib.import_module(self._module_name)
        self._source = Path(module.__file__).read_text(encoding="utf-8")
        self._live_source = self._source
        self._original_class = getattr(self._environment, self._class_name)
        self._candidate_module_name = f"tau2.domains.{domain}._ato_candidate_tools"

    def extract(self) -> ToolSet:
        return ToolSet({"tools.py": self._source}, ("tools.py",), self.language_rules)

    def effective_toolset(self) -> ToolSet:
        return ToolSet(
            {"tools.py": self._live_source}, ("tools.py",), self.language_rules
        )

    def validator(self) -> Validator:
        if self.descriptions_only:
            return Tau2DescriptionValidator(self.kind, ("tools.py",), self._source)
        return Tau2CodeValidator(
            self.kind, ("tools.py",), self._source, self._class_name
        )

    def apply(self, candidate: Candidate) -> None:
        result = self.validator().validate(candidate)
        if not result.ok:
            raise ValueError(result.log)
        source = candidate.files.get("tools.py")
        if source is None:
            return
        self.restore()
        module = types.ModuleType(self._candidate_module_name)
        module.__package__ = f"tau2.domains.{self.domain}"
        sys.modules[self._candidate_module_name] = module
        try:
            exec(compile(source, self._candidate_module_name, "exec"), module.__dict__)
            candidate_class = getattr(module, self._class_name)
        except BaseException:
            sys.modules.pop(self._candidate_module_name, None)
            raise
        setattr(self._environment, self._class_name, candidate_class)
        self._live_source = source

    def restore(self) -> None:
        setattr(self._environment, self._class_name, self._original_class)
        sys.modules.pop(self._candidate_module_name, None)
        self._live_source = self._source


class Tau2Agent(Agent):
    def __init__(self, model: str, *, user_model: str) -> None:
        self.id = model
        self.model = model
        self.user_model = user_model


class Tau2Benchmark(Benchmark):
    def __init__(
        self,
        domain: str,
        *,
        train_tasks: list[str],
        test_tasks: list[str],
        max_steps: int = 30,
    ) -> None:
        if domain not in _DOMAINS:
            raise ValueError(f"unsupported tau2 domain: {domain}")
        if set(train_tasks) & set(test_tasks):
            raise ValueError("train and test tasks overlap")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.name = f"tau2-{domain}"
        self.domain = domain
        self._splits = {"train": tuple(train_tasks), "test": tuple(test_tasks)}
        self.max_steps = max_steps

    def tasks(self, split: str) -> list[str]:
        return list(self._splits[split])

    def evaluate(self, agent: Agent, tasks: list[str], tools: ToolSet) -> RunResult:
        from tau2.run import EvaluationType, get_tasks, run_tasks

        if not isinstance(agent, Tau2Agent):
            raise TypeError("Tau2Benchmark requires Tau2Agent")
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("provide distinct task IDs")
        if not set(tasks) <= set(self._splits["train"] + self._splits["test"]):
            raise ValueError("task outside configured splits")
        results = run_tasks(
            domain=self.domain,
            tasks=get_tasks(self.domain, task_ids=tasks),
            agent="llm_agent",
            user="user_simulator",
            llm_agent=agent.model,
            llm_user=agent.user_model,
            llm_args_agent={"temperature": 0.0},
            llm_args_user={"temperature": 0.0},
            num_trials=1,
            max_steps=self.max_steps,
            save_to=None,
            console_display=False,
            evaluation_type=EvaluationType.ALL,
            max_concurrency=1,
        )
        runs = []
        for sim in results.simulations:
            values = [
                usd(getattr(sim, name, None)) for name in ("agent_cost", "user_cost")
            ]
            messages = [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in sim.messages or []
            ]
            runs.append(
                TaskRun(
                    task_id=str(sim.task_id),
                    reward=float(getattr(sim.reward_info, "reward", 0.0) or 0.0),
                    trajectory=messages,
                    cost_usd=total(values),
                    known_cost_usd=sum(v for v in values if v is not None),
                )
            )
        if sorted(r.task_id for r in runs) != sorted(tasks):
            raise RuntimeError("tau2 returned incomplete task results")
        return RunResult(self.name, agent.id, tuple(runs))
