# Public benchmark checkouts

The benchmark and agent repositories are separate from `agent-tool-opt-core`. Clone them next to this repository; their code and datasets are not redistributed here.

| Evaluation | Public upstream | Local checkout | Purpose |
|---|---|---|---|
| TauBench Verified | [amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified) | `../tau2-bench-verified` | Airline, retail, and telecom simulations and tools |
| TerminalBench 2 | [laude-institute/terminal-bench-2](https://github.com/laude-institute/terminal-bench-2) | `../terminal-bench-2` | Terminal tasks and verifiers |
| OpenThoughts TBLite | [open-thoughts/OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite) | `../OpenThoughts-TBLite` | Smaller terminal task set |
| OpenCode | [anomalyco/opencode](https://github.com/anomalyco/opencode) | `../opencode` | Agent tool descriptions and implementation for both terminal evaluations |
| Harbor | [laude-institute/harbor](https://github.com/laude-institute/harbor) | `../harbor` | Local-Docker terminal task runner |

```bash
git clone https://github.com/amazon-agi/tau2-bench-verified.git ../tau2-bench-verified
git clone https://github.com/laude-institute/terminal-bench-2.git ../terminal-bench-2
git clone https://github.com/open-thoughts/OpenThoughts-TBLite.git ../OpenThoughts-TBLite
git clone https://github.com/anomalyco/opencode.git ../opencode
git clone https://github.com/laude-institute/harbor.git ../harbor
```

Install the upstream runners from their local checkouts in the same environment as this package. Harbor currently requires Python 3.12 or newer.

```bash
python -m pip install -e ../tau2-bench-verified
python -m pip install -e ../harbor
```

The terminal adapters use Harbor's `--path` mode against the cloned task directories, not the remote dataset registry. Create a JSON split manifest with task directory names, for example:

```json
{"train": ["task-a"], "test": ["task-b"]}
```

The terminal agent must run OpenCode **from source**: a prebuilt OpenCode binary can embed its tool text and silently ignore an edited description. On a Linux machine with Bun installed, install the OpenCode checkout's pinned dependencies and make a source bundle with symlinks dereferenced:

```bash
cd ../opencode
bun install --frozen-lockfile
tar -chf ../opencode-source-linux.tar package.json bun.lock packages node_modules
cd ../agent-tool-opt-core
```

Pass that tarball and the absolute path to a Linux Bun executable as `source_bundle` and `bun_linux_binary`. The bundle must contain the root `package.json`, `packages/opencode/src/index.ts`, and `node_modules`. The custom Harbor agent checks these inputs, uploads them to each local-Docker sandbox, overlays the baseline or candidate tool files, then starts OpenCode. It does not run a package manager or download an installer in the sandbox. Configure model credentials through Harbor/OpenCode outside this repository; do not put them in the split manifest or source bundle.

Both tool targets default to description-only optimization. To port the original adapters' full-code scope, construct either target with `descriptions_only=False`; TauBench then allows implementation edits while preserving `@is_tool` names, decorators, and signatures, and OpenCode exposes active `.ts` modules alongside `.txt` descriptions. The TypeScript gate requires Bun on the host and parse-checks changed modules. Full-code candidates execute generated code: use a disposable, isolated environment and scoped model credentials, especially for TauBench's in-process class swap.

The adapters implement the core `Benchmark`, `Agent`, and `ToolTarget` interfaces. For example:

```python
from pathlib import Path
from agent_tool_opt_core.adapters.tau2 import Tau2Agent, Tau2Benchmark, Tau2ToolTarget
from agent_tool_opt_core.adapters.terminal import (
    OpenCodeAgent,
    OpenCodeToolTarget,
    TerminalBench2,
)
from agent_tool_opt_core.driver import optimize
from agent_tool_opt_core.optimizers.pi import PiOptimizer

# Choose and freeze disjoint TauBench task IDs before running.
tau_benchmark = Tau2Benchmark(
    "airline", train_tasks=["train-id"], test_tasks=["test-id"]
)
tau_agent = Tau2Agent("agent-model", user_model="user-simulator-model")
tau_target = Tau2ToolTarget("airline")

# Or use TerminalBench2; OpenThoughtsTBLite has the same constructor.
terminal_benchmark = TerminalBench2(
    benchmark_checkout=Path("../terminal-bench-2"),
    split_manifest=Path("splits/terminal.json"),
    source_bundle=Path("../opencode-source-linux.tar"),
    bun_linux_binary=Path("/absolute/path/to/linux/bun"),
    jobs_dir=Path("runs/harbor"),
)
terminal_agent = OpenCodeAgent("provider/agent-model")
terminal_target = OpenCodeToolTarget(Path("../opencode"))

optimizer = PiOptimizer(model="optimizer-model", provider="optimizer-provider")
candidate, test_metrics = optimize(
    terminal_benchmark,
    terminal_agent,
    terminal_target,
    optimizer,
    train=terminal_benchmark.tasks("train"),
    test=terminal_benchmark.tasks("test"),
    scratch=Path("runs/optimizer"),
)
```

Replace the illustrative task/model values with actual IDs and configured provider models. Run `optimize` with the TauBench triplet instead to optimize TauBench descriptions. Keep baseline and candidate evaluations on the same task IDs, agent model, and runtime. Terminal runs currently use one Harbor trial per task. The adapters are unit-tested with a mocked runner, but live Harbor and TauBench executions have not yet been verified; run a small smoke task before using them for a study.

The benchmark projects change independently. Record each checkout's Git commit alongside results and confirm its own dependency and license terms before distributing any derived artifacts.
