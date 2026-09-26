"""Five-phase experiment contract shared by the public benchmark adapters."""

from __future__ import annotations

import json

import pytest

from agent_tool_opt_core.adapters.run_phases import run_five_phases
from agent_tool_opt_core.adapters.run_tau2 import main as tau2_main
from agent_tool_opt_core.adapters.run_terminal import main as terminal_main
from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Optimizer,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
)


class ExampleAgent(Agent):
    id = "same-agent-model"


class ExampleTarget(ToolTarget):
    kind = "txt"
    language_rules = "Improve tool.txt"

    def __init__(self):
        self.baseline = ToolSet(
            {"tool.txt": "baseline"}, ("tool.txt",), self.language_rules
        )
        self.live = self.baseline

    def extract(self):
        return self.baseline

    def effective_toolset(self):
        return self.live

    def apply(self, candidate):
        if not self.validator().validate(candidate).ok:
            raise ValueError("invalid candidate")
        self.live = ToolSet(
            {**self.baseline.files, **candidate.files},
            self.baseline.allowlist,
            self.language_rules,
        )

    def restore(self):
        self.live = self.baseline


class ExampleBenchmark(Benchmark):
    name = "example"

    def __init__(self):
        self.calls = []

    def tasks(self, split):
        return ["train-1"] if split == "train" else ["test-1"]

    def evaluate(self, agent, tasks, tools):
        self.calls.append((tuple(tasks), tools.files["tool.txt"]))
        edited = tools.files["tool.txt"] == "edited"
        return RunResult(
            self.name,
            agent.id,
            tuple(
                TaskRun(
                    task_id=name,
                    reward=float(edited),
                    trajectory={"task": name, "tool": tools.files["tool.txt"]},
                    cost_usd=0.2 if edited else 0.1,
                )
                for name in tasks
            ),
        )


class ExampleOptimizer(Optimizer):
    id = "example-optimizer"

    def __init__(self, output_dir, candidate=None, *, wants_train_eval=False):
        self.output_dir = output_dir
        self.candidate = candidate or Candidate({"tool.txt": "edited"})
        self.wants_train_eval = wants_train_eval
        self.seen = []

    def propose(self, tools, run, validate, scratch, train_eval=None):
        self.seen = [(task.task_id, task.trajectory) for task in run.runs]
        assert run.split == "train"
        assert not (self.output_dir / "baseline_test.json").exists()
        if self.wants_train_eval:
            assert train_eval.train_tasks == ("train-1",)
            train_eval.evaluate(self.candidate)
            with pytest.raises(ValueError, match="outside train split"):
                train_eval.evaluate(self.candidate, ["test-1"])
        else:
            assert train_eval is None
        return self.candidate


def _run(tmp_path, optimizer):
    benchmark = ExampleBenchmark()
    target = ExampleTarget()
    result = run_five_phases(
        benchmark,
        ExampleAgent(),
        target,
        optimizer,
        train=benchmark.tasks("train"),
        test=benchmark.tasks("test"),
        output_dir=tmp_path / "output",
    )
    return result, benchmark, target


def test_five_phases_save_paired_results_and_restore_tools(tmp_path):
    output = tmp_path / "output"
    optimizer = ExampleOptimizer(output)
    result, benchmark, target = _run(tmp_path, optimizer)
    assert benchmark.calls == [
        (("train-1",), "baseline"),
        (("test-1",), "baseline"),
        (("train-1",), "edited"),
        (("test-1",), "edited"),
    ]
    assert [task_id for task_id, _ in optimizer.seen] == ["train-1"]
    assert result["optimization"]["status"] == "candidate_applied"
    assert result["phases"]["baseline_test"]["avg_reward"] == 0.0
    assert result["phases"]["optimized_test"]["avg_reward"] == 1.0
    assert result["phases"]["baseline_test"]["cost_usd"] == 0.1
    assert result["phases"]["optimized_test"]["cost_usd"] == 0.2
    assert result["comparison"]["test"]["avg_reward_delta"] == 1.0
    assert result["comparison"]["test"]["cost_usd_delta"] == pytest.approx(0.1)
    assert target.effective_toolset() == target.extract()
    for name in (
        "baseline_train",
        "baseline_test",
        "optimized_train",
        "optimized_test",
    ):
        assert (output / f"{name}.json").is_file()
    assert "runs" not in result["phases"]["baseline_train"]
    assert (
        json.loads((output / "baseline_train.json").read_text())["runs"][0][
            "trajectory"
        ]["task"]
        == "train-1"
    )
    assert (output / "candidate" / "tool.txt").read_text() == "edited"
    assert json.loads((output / "summary.json").read_text()) == result


def test_no_edit_skips_optimized_phases(tmp_path):
    optimizer = ExampleOptimizer(
        tmp_path / "output", Candidate({"tool.txt": "baseline"})
    )
    result, benchmark, target = _run(tmp_path, optimizer)
    assert result["optimization"]["status"] == "no_edit"
    assert len(benchmark.calls) == 2
    assert not (tmp_path / "output" / "optimized_test.json").exists()
    assert target.effective_toolset() == target.extract()


def test_train_evaluator_never_accepts_test_tasks(tmp_path):
    optimizer = ExampleOptimizer(tmp_path / "output", wants_train_eval=True)
    result, benchmark, _ = _run(tmp_path, optimizer)
    assert result["optimizer_cost"]["train_search"]["known_cost_usd"] == 0.2
    assert benchmark.calls[2][0] == ("train-1",)


def test_output_directory_must_be_new(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    benchmark = ExampleBenchmark()
    with pytest.raises(FileExistsError):
        run_five_phases(
            benchmark,
            ExampleAgent(),
            ExampleTarget(),
            ExampleOptimizer(output),
            train=["train-1"],
            test=["test-1"],
            output_dir=output,
        )
    assert benchmark.calls == []


def test_optimizer_failure_keeps_baselines_without_scoring_candidate(tmp_path):
    class FailingOptimizer(ExampleOptimizer):
        def propose(self, tools, run, validate, scratch, train_eval=None):
            raise RuntimeError("private provider error detail")

    benchmark = ExampleBenchmark()
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="private provider error detail"):
        run_five_phases(
            benchmark,
            ExampleAgent(),
            ExampleTarget(),
            FailingOptimizer(output),
            train=["train-1"],
            test=["test-1"],
            output_dir=output,
        )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["optimization"] == {
        "status": "optimizer_failed",
        "error_type": "RuntimeError",
    }
    assert "private provider error detail" not in (output / "summary.json").read_text()
    assert (output / "baseline_test.json").is_file()
    assert len(benchmark.calls) == 2


def test_candidate_runtime_failure_is_not_scored_as_a_loss(tmp_path):
    class FailingBenchmark(ExampleBenchmark):
        def evaluate(self, agent, tasks, tools):
            if tasks == ["test-1"] and tools.files["tool.txt"] == "edited":
                raise RuntimeError("private runtime detail")
            return super().evaluate(agent, tasks, tools)

    benchmark = FailingBenchmark()
    target = ExampleTarget()
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="private runtime detail"):
        run_five_phases(
            benchmark,
            ExampleAgent(),
            target,
            ExampleOptimizer(output),
            train=["train-1"],
            test=["test-1"],
            output_dir=output,
        )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["optimization"]["status"] == "evaluation_failed"
    assert summary["optimization"]["failed_phase"] == "optimized_test"
    assert "private runtime detail" not in (output / "summary.json").read_text()
    assert (output / "optimized_train.json").is_file()
    assert not (output / "optimized_test.json").exists()
    assert target.effective_toolset() == target.extract()


def test_baseline_failure_has_no_optimized_result(tmp_path):
    class FailingBenchmark(ExampleBenchmark):
        def evaluate(self, agent, tasks, tools):
            if tasks == ["test-1"]:
                raise RuntimeError("private baseline detail")
            return super().evaluate(agent, tasks, tools)

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="private baseline detail"):
        run_five_phases(
            FailingBenchmark(),
            ExampleAgent(),
            ExampleTarget(),
            ExampleOptimizer(output),
            train=["train-1"],
            test=["test-1"],
            output_dir=output,
        )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["optimization"]["status"] == "baseline_failed"
    assert summary["optimization"]["failed_phase"] == "baseline_test"
    assert "private baseline detail" not in (output / "summary.json").read_text()
    assert (output / "baseline_train.json").is_file()
    assert not (output / "optimized_test.json").exists()


def test_baseline_only_can_be_reused_without_new_baseline_calls(tmp_path):
    benchmark = ExampleBenchmark()
    agent = ExampleAgent()
    target = ExampleTarget()
    baseline_dir = tmp_path / "baseline"
    baseline = run_five_phases(
        benchmark,
        agent,
        target,
        None,
        train=["train-1"],
        test=["test-1"],
        output_dir=baseline_dir,
    )
    assert baseline["optimization"]["status"] == "baseline_only"
    assert len(benchmark.calls) == 2
    assert (baseline_dir / "report.html").is_file()

    output = tmp_path / "optimized"
    reused = run_five_phases(
        benchmark,
        agent,
        target,
        ExampleOptimizer(output),
        train=["train-1"],
        test=["test-1"],
        output_dir=output,
        baseline_dir=baseline_dir,
    )
    assert reused["baseline_source"] == str(baseline_dir.resolve())
    assert benchmark.calls[2:] == [
        (("train-1",), "edited"),
        (("test-1",), "edited"),
    ]
    assert reused["comparison"]["test"]["avg_reward_delta"] == 1.0


def test_baseline_reuse_rejects_changed_tool_snapshot(tmp_path):
    benchmark = ExampleBenchmark()
    baseline_dir = tmp_path / "baseline"
    run_five_phases(
        benchmark,
        ExampleAgent(),
        ExampleTarget(),
        None,
        train=["train-1"],
        test=["test-1"],
        output_dir=baseline_dir,
    )
    changed_target = ExampleTarget()
    changed_target.baseline = ToolSet(
        {"tool.txt": "different source"}, ("tool.txt",), changed_target.language_rules
    )
    changed_target.live = changed_target.baseline
    output = tmp_path / "invalid-reuse"
    with pytest.raises(ValueError, match="incompatible"):
        run_five_phases(
            benchmark,
            ExampleAgent(),
            changed_target,
            ExampleOptimizer(output),
            train=["train-1"],
            test=["test-1"],
            output_dir=output,
            baseline_dir=baseline_dir,
        )
    assert not output.exists()


def test_baseline_reuse_rejects_incomplete_artifact(tmp_path):
    baseline_dir = tmp_path / "baseline"
    benchmark = ExampleBenchmark()
    run_five_phases(
        benchmark,
        ExampleAgent(),
        ExampleTarget(),
        None,
        train=["train-1"],
        test=["test-1"],
        output_dir=baseline_dir,
    )
    artifact_path = baseline_dir / "baseline_train.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["runs"] = []
    artifact_path.write_text(json.dumps(artifact))
    output = tmp_path / "reused"
    with pytest.raises(ValueError, match="incomplete"):
        run_five_phases(
            benchmark,
            ExampleAgent(),
            ExampleTarget(),
            ExampleOptimizer(output),
            train=["train-1"],
            test=["test-1"],
            output_dir=output,
            baseline_dir=baseline_dir,
        )
    assert not output.exists()


def test_multiple_candidates_each_get_independent_train_test_results(tmp_path):
    class SequenceOptimizer(ExampleOptimizer):
        def __init__(self, output_dir):
            super().__init__(output_dir)
            self.n = 0

        def propose(self, tools, run, validate, scratch, train_eval=None):
            assert run.split == "train"
            assert not (self.output_dir / "baseline_test.json").exists()
            text = "edited" if self.n == 0 else "different edit"
            self.n += 1
            return Candidate({"tool.txt": text})

    output = tmp_path / "output"
    optimizer = SequenceOptimizer(output)
    benchmark = ExampleBenchmark()
    result = run_five_phases(
        benchmark,
        ExampleAgent(),
        ExampleTarget(),
        optimizer,
        train=["train-1"],
        test=["test-1"],
        output_dir=output,
        num_candidates=2,
    )
    assert optimizer.n == 2
    assert result["optimization"]["completed"] == 2
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["comparison"]["test"]["avg_reward_delta"] == 1.0
    assert result["candidates"][1]["comparison"]["test"]["avg_reward_delta"] == 0.0
    assert (output / "candidate_00" / "optimized_test.json").is_file()
    assert (output / "candidate_01" / "optimized_test.json").is_file()
    assert (output / "candidate_00" / "candidate" / "tool.txt").read_text() == "edited"
    assert "candidate 0" in (output / "report.html").read_text()
    assert "Optimizer LLM USD" in (output / "report.html").read_text()


def test_multiple_no_edit_candidates_have_no_efficacy_result(tmp_path):
    output = tmp_path / "output"
    optimizer = ExampleOptimizer(output, Candidate({"tool.txt": "baseline"}))
    result = run_five_phases(
        ExampleBenchmark(),
        ExampleAgent(),
        ExampleTarget(),
        optimizer,
        train=["train-1"],
        test=["test-1"],
        output_dir=output,
        num_candidates=2,
    )
    assert result["optimization"] == {
        "status": "no_valid_candidate",
        "requested": 2,
        "completed": 0,
    }
    assert not (output / "candidate_00" / "optimized_test.json").exists()
    assert not (output / "candidate_01" / "optimized_test.json").exists()


def test_parallel_phases_use_immutable_toolsets(tmp_path):
    from threading import Barrier

    class ParallelBenchmark(ExampleBenchmark):
        parallel_safe = True
        n_concurrent = 2

        def __init__(self):
            super().__init__()
            self.barrier = Barrier(2)

        def evaluate_parallel(self, agent, tasks, tools):
            self.barrier.wait(timeout=2)
            return super().evaluate(agent, tasks, tools)

    benchmark = ParallelBenchmark()
    target = ExampleTarget()
    result = run_five_phases(
        benchmark,
        ExampleAgent(),
        target,
        ExampleOptimizer(tmp_path / "output"),
        train=["train-1"],
        test=["test-1"],
        output_dir=tmp_path / "output",
        parallel_phases=True,
    )
    assert result["comparison"]["test"]["avg_reward_delta"] == 1.0
    assert target.effective_toolset() == target.extract()
    assert len(benchmark.calls) == 4


def test_html_report_escapes_external_names(tmp_path):
    benchmark = ExampleBenchmark()
    benchmark.name = "<script>alert(1)</script>"
    output = tmp_path / "output"
    run_five_phases(
        benchmark,
        ExampleAgent(),
        ExampleTarget(),
        None,
        train=["train-1"],
        test=["test-1"],
        output_dir=output,
    )
    page = (output / "report.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


@pytest.mark.parametrize("entrypoint", [tau2_main, terminal_main])
def test_public_runner_has_help_without_upstream_runtime(entrypoint, capsys):
    with pytest.raises(SystemExit) as exit_info:
        entrypoint(["--help"])
    assert exit_info.value.code == 0
    assert "--num-trials" in capsys.readouterr().out
