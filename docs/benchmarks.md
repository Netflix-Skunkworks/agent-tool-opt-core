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

The terminal agent must run OpenCode **from source**: a prebuilt OpenCode binary can embed its tool text and silently ignore an edited description. On a Linux machine, use the Bun version declared by the checkout's `packageManager` (1.3.14 for the version tested here), and provide Python 3, `make`, and a C compiler for native dependency builds. Configure Bun to use your approved package registry. Install the checkout's pinned dependencies and make a source bundle while preserving its workspace links:

```bash
cd ../opencode
bun install --frozen-lockfile
tar -cf ../opencode-source-linux.tar package.json bun.lock packages node_modules
cd ../agent-tool-opt-core
```

Pass that tarball and the absolute path to a Linux Bun executable as `source_bundle` and `bun_linux_binary`. The bundle must contain the root `package.json`, `packages/opencode/src/index.ts`, and `node_modules`. Before a run, the adapter checks that the bundle's package manifests, lockfile, entry point, and tool files match the local OpenCode checkout; archive links must remain within the bundle. Do not use `tar -h`: dereferencing Bun's workspace links can inflate the archive many-fold. The custom Harbor agent uploads the bundle to each local-Docker sandbox, overlays the baseline or candidate tool files, then starts OpenCode. It does not run a package manager or download an installer in the sandbox. Configure model credentials through Harbor/OpenCode outside this repository; do not put them in the split manifest or source bundle.

Both tool targets default to description-only optimization. TauBench presents `descriptions.md` as the editable file, with its full `tools.py` and domain policy as read-only context; description edits are spliced into docstrings and checked against TauBench's tool schemas. Pass `--scope full` to enable the original adapters' full-code scope: TauBench permits implementation edits while preserving `@is_tool` contracts, and OpenCode exposes active `.ts` modules alongside `.txt` descriptions. The TypeScript gate requires Bun on the host and parse-checks changed modules. Full-code candidates execute generated code: use a disposable, isolated environment and scoped model credentials, especially for TauBench's in-process class swap.

The public runners keep the original five phases: baseline train, baseline test, optimize on train transcripts only, optimized train, and optimized test. With frozen split JSON files and real model names, invoke them as follows:

```bash
ato-tau2 \
  --domain airline --split splits/airline.json \
  --agent-model agent-model --user-model user-simulator-model \
  --optimizer pi --optimizer-model optimizer-model \
  --methods reward_shaping,generalization --num-trials 3 \
  --out runs/airline-001

ato-terminal \
  --benchmark tb2 --benchmark-checkout ../terminal-bench-2 \
  --opencode-checkout ../opencode --split splits/tb2.json \
  --source-bundle ../opencode-source-linux.tar \
  --bun-linux-binary /absolute/path/to/linux/bun \
  --jobs-dir runs/harbor --agent-model provider/agent-model \
  --optimizer pi --optimizer-model optimizer-model \
  --methods reward_shaping,generalization --num-trials 3 \
  --n-concurrent 4 --parallel-phases \
  --out runs/tb2-001
```

For TBLite, use `--benchmark tblite` and `--benchmark-checkout ../OpenThoughts-TBLite`. Substitute configured model IDs, provider credentials, and actual task IDs; `--out` must name a new directory. Terminal `--n-concurrent` is the total task concurrency budget; with `--parallel-phases`, each of the two simultaneous jobs receives at most half (minimum total: two). TauBench phases remain sequential because its tool-class swap is process-global.

The `ato-tau2` and `ato-terminal` commands are installed with this package. Equivalent module invocations are `python -m agent_tool_opt_core.adapters.run_tau2` and `python -m agent_tool_opt_core.adapters.run_terminal`.

To capture a baseline once and reuse it for several optimization attempts, run either CLI with `--baseline-only --out runs/baseline-001` (no `--optimizer-model` required). Later pass `--baseline-dir runs/baseline-001 --num-candidates 3 --out runs/candidates-001` with the same benchmark, agent, split and trial count. Reuse checks these fields and the editable tool/context snapshot before any new run. Each candidate is proposed independently from the original training baseline; candidate outputs are saved under `candidate_00/`, `candidate_01/`, and so on. Pin benchmark checkout commits too: local uncommitted task changes are not captured by the reuse check.

Each run saves `baseline_train.json`, `baseline_test.json`, `summary.json`, and a static `report.html`. Successful single-candidate runs also save `optimized_train.json`, `optimized_test.json` and candidate files at the top level; multi-candidate runs save these under each candidate directory. A no-edit or failed candidate is reported without a misleading optimized score. Phase files contain task reward and raw transcript; the summary and HTML report contain metrics only. Known/unknown benchmark-reported model cost and optimizer cost are separate; Docker compute and other infrastructure charges are not estimated. Keep task IDs, models, trial count, and runtime fixed across the A/B arms. All candidate proposals receive only training transcripts, and the held-out baseline file is not written until every proposal returns. For strict filesystem isolation, configure `PiOptimizer.sandbox_cmd` through the Python API or run the CLI in an isolated environment. Offline adapter tests and a paid one-train/one-test TauBench A/B smoke have completed; the latter verified real model calls and candidate application, not an efficacy improvement or a full benchmark run. TerminalBench 2 and TBLite have not been run end-to-end with real model calls.

For a paid end-to-end smoke, use one real train task and one real test task with `--num-trials 1`. Run `--baseline-only` first, then reuse it with `--baseline-dir` and an optimizer. Confirm `summary.json` reports `candidate_applied`, all four phase files exist, the candidate file differs from the baseline, and no trial reports an infrastructure or authentication error. A `no_edit` proposal does not verify candidate injection. Never store API keys in the repository or result files. The completed TauBench smoke does not establish improvement: task rewards on a one-task split are noisy, and the optimized arm must be evaluated on larger frozen splits before drawing conclusions.

The benchmark projects change independently. Record each checkout's Git commit alongside results and confirm its own dependency and license terms before distributing any derived artifacts.
