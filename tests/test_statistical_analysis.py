"""Reward/cost CLI parity, shared inference, and cost validation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_tool_opt_core.analysis import cost, paired, reward

SRC = Path(__file__).resolve().parents[1] / "src"


def write_results(path, agent_cost=0.2, reward=1, tasks=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": str(task),
                        "trial": trial,
                        "agent_cost": agent_cost,
                        "user_cost": 0.1,
                        "reward_info": {"reward": reward},
                    }
                    for task in range(tasks)
                    for trial in range(6)
                ]
            }
        )
    )
    return path


def test_cost_uses_task_clusters_and_both_cost_components(tmp_path):
    a = write_results(tmp_path / "a.json")
    b = write_results(tmp_path / "b.json", agent_cost=0.3)
    means_a, trials_a = cost.per_task_costs(a)
    means_b, trials_b = cost.per_task_costs(b)
    assert trials_a == trials_b
    assert means_a["0"] == pytest.approx(0.3)
    assert means_b["0"] == pytest.approx(0.4)
    # Four task clusters, not 24 independent trials: two extremes / 2**4.
    assert paired.paired_permutation_p(means_a, means_b) == 0.125
    assert paired.paired_mean_ci(means_a, means_b, 100, 300) == pytest.approx(
        (0.1, 0.1)
    )


@pytest.mark.parametrize("amount", [None, float("nan"), float("inf"), -1, "0.2", True])
def test_invalid_cost_rejected(tmp_path, amount):
    path = write_results(tmp_path / "invalid.json", agent_cost=amount)
    with pytest.raises(ValueError, match="missing/invalid cost"):
        cost.per_task_costs(path)


def test_zero_cost_allowed_but_missing_component_rejected(tmp_path):
    path = write_results(tmp_path / "a.json", agent_cost=0)
    assert cost.per_task_costs(path)[0]["0"] == pytest.approx(0.1)
    data = json.loads(path.read_text())
    del data["simulations"][0]["user_cost"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing/invalid cost"):
        cost.per_task_costs(path)


def test_duplicate_and_unmatched_trials_rejected(tmp_path):
    a = write_results(tmp_path / "a.json")
    b = write_results(tmp_path / "b.json")
    data = json.loads(b.read_text())
    data["simulations"].pop()
    b.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical task/trial coverage"):
        cost.compare(a, b, "A", "B", 100, 100, 300)
    data["simulations"].append(data["simulations"][0])
    b.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate trial"):
        cost.per_task_costs(b)


def test_shared_inference_is_strict_and_monte_carlo_is_deterministic():
    a = {str(i): 0.0 for i in range(23)}
    b = dict.fromkeys(a, 1.0)
    # With 23 tasks the MC path is used; no sampled assignment is as extreme.
    assert paired.paired_permutation_p(a, b, n_perm=100, seed=300) == 1 / 101
    assert paired.paired_permutation_p(a, a, n_perm=100, seed=300) == 1.0
    with pytest.raises(ValueError, match="identical nonempty task sets"):
        paired.paired_estimate({"a": 0.0}, {"b": 1.0}, n_perm=100, n_boot=100, seed=300)


@pytest.mark.parametrize("selection", ["run", "pair", "ours"])
@pytest.mark.parametrize("module", ["reward", "cost"])
def test_cli_selection_modes_and_harness_filenames(tmp_path, selection, module):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_phase = "baseline" if selection == "run" else "optimized"
    write_results(a_dir / f"{a_phase}_test_c00.json", reward=0)
    write_results(b_dir / "optimized_test_c00.json", agent_cost=0.1)
    if selection == "run":
        write_results(a_dir / "optimized_test_c00.json", agent_cost=0.1)
        args = ["--run", str(a_dir)]
    elif selection == "pair":
        args = ["--a", str(a_dir), "--b", str(b_dir)]
    else:
        args = ["--ours", str(b_dir), "--baselines", str(a_dir)]
    args += ["--n-boot", "100", "--n-perm", "100", "--seed", "300"]
    if module == "reward":
        args += ["--ks", "1"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-m", f"agent_tool_opt_core.analysis.{module}", *args],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "exact" in completed.stdout
    if module == "reward":
        assert "0.1250" in completed.stdout
        assert "pass^k(A)" in completed.stdout
    else:
        assert "delta=-0.10000000" in completed.stdout
        assert "perm p=0.125000" in completed.stdout


def test_reward_analysis_still_accepts_missing_cost(tmp_path, capsys):
    a = write_results(tmp_path / "a.json", agent_cost=None)
    reward.compare(a, a, "A", "A", [1], 100, 100, 300)
    assert "1.0000" in capsys.readouterr().out
