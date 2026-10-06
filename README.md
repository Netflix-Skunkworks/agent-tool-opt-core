# AT-AT Tool Optimization

This repository runs AT-AT experiments on agent tools: collect baseline task
transcripts and rewards, have Pi propose tool edits with reward shaping and
generalization, validate those edits, and compare the same agent on frozen
training and test tasks. It also supports other optimizers for comparisons.
Here reward shaping supplies raw training transcripts grouped by task reward;
generalization asks Pi to favor edits that could transfer to held-out tasks.

> **Status:** pre-release research code. The CLIs and benchmark integrations may change.

## Features

- a runnable Pi + reward-shaping + generalization AT-AT workflow;
- Pi and one-shot LLM in the benchmark CLIs; DRAFT, GEPA, and ToolObserver
  through the Python optimizer catalog;
- common interfaces for benchmarks, agents, tool targets, and validators;
- train/test separation and paired evaluation utilities;
- normalized provider usage, estimated cost, and provenance metadata; and
- an optional Metaflow harness for TauBench, with local file output or
  datastore artifacts for remote workers.

## Set up the repository

This is a runnable repository, not a published PyPI package. Use Python 3.12
or newer if you plan to run all three benchmark adapters (Harbor requires
3.12+). Clone the repo and install it in editable mode so its CLIs and imports
are available in your environment:

```bash
git clone https://github.com/Netflix-Skunkworks/agent-tool-opt-core.git
cd agent-tool-opt-core
python -m pip install -e .
```

### Pi CLI for full AT-AT

The Python install does not include Pi. Install Node.js 22.19+ and the external
`pi` CLI:

```bash
npm install -g @earendil-works/pi-coding-agent@0.84.1
pi --version
```

By default, Pi can read absolute host paths and inherits the process environment.
On Linux, add `--pi-sandbox bubblewrap --pi-sandbox-env OPENAI_API_KEY` to a Pi
run to restrict its filesystem access and forward only the selected provider
variable. See [Pi sandbox setup and boundaries](docs/pi_sandbox.md) for
prerequisites, other providers, and Metaflow usage. Use scoped credentials and
an isolated worker for generated-code execution. Pi is not needed for the
synthetic smoke or the one-shot LLM optimizer.

The `metaflow` extra installs upstream Metaflow. The harness runs the TauBench
five-phase loop: baseline train/test, train-only optimization, then candidate
train/test evaluation. It supports local file output and a backend-neutral
datastore mode for configured remote workers; see
[remote Metaflow setup](docs/metaflow_remote.md). TerminalBench 2 and TBLite
use their separate CLI adapter, not this Metaflow harness.

To smoke-test the real flow graph without a model key:

```bash
python -m pip install -e ".[metaflow]"
python harness/run_harness_metaflow.py show
python harness/run_harness_metaflow.py run --benchmark synthetic --output-dir runs/metaflow-smoke-001
```

The synthetic mode makes no model calls. Output directories must be new, so
choose a different suffix for each rerun.

## Benchmark source checkouts

The benchmark datasets and agent code are not bundled with this repository. For
TauBench Verified, clone and install its upstream repository next to this one:

```bash
git clone https://github.com/amazon-agi/tau2-bench-verified.git ../tau2-bench-verified
python -m pip install -e ../tau2-bench-verified
```

For TerminalBench 2 or OpenThoughts TBLite, also clone their task repositories,
OpenCode, and Harbor:

```bash
git clone https://github.com/laude-institute/terminal-bench-2.git ../terminal-bench-2
git clone https://github.com/open-thoughts/OpenThoughts-TBLite.git ../OpenThoughts-TBLite
git clone https://github.com/anomalyco/opencode.git ../opencode
git clone https://github.com/laude-institute/harbor.git ../harbor
python -m pip install -e ../harbor
```

The terminal adapters run tasks through Harbor, with OpenCode source and edited
tool files uploaded into each sandbox. Terminal runs also require a Linux
OpenCode dependency bundle and Linux Bun executable; see [benchmark setup](docs/benchmarks.md).

## Live TauBench run with Metaflow

The commands below use local file output. For remote workers, use
`--artifact-only` and `--baseline-run-id` as shown in the
[remote setup guide](docs/metaflow_remote.md).

The included [airline smoke split](examples/airline-smoke-split.json) selects one
training task and one different test task from the upstream airline
`data/tau2/domains/airline/split_tasks.json`. Check the IDs against your pinned
TauBench checkout before running. This small split tests the integration, not
optimization efficacy. For a full experiment, provide your own frozen JSON
manifest with nonempty, disjoint `train` and `test` task-ID arrays.

Supply `OPENAI_API_KEY` through your shell or secret manager without putting
the value in a command, the repository, or a result file. The commands below
make paid model calls. Other providers can be configured through LiteLLM; use
model IDs and credentials supported by that provider.

First capture a reusable baseline:

```bash
python harness/run_harness_metaflow.py run \
  --benchmark tau2 --domain airline --split examples/airline-smoke-split.json \
  --agent-model gpt-4o-mini --user-model gpt-4o-mini \
  --num-trials 1 --max-steps 8 --skip-optimize \
  --output-dir runs/airline-baseline-001
```

With Pi installed, run reward shaping and generalization using the same frozen
baseline and agent settings:

```bash
python harness/run_harness_metaflow.py run \
  --benchmark tau2 --domain airline --split examples/airline-smoke-split.json \
  --agent-model gpt-4o-mini --user-model gpt-4o-mini \
  --num-trials 1 --max-steps 8 \
  --optimizer pi --optimizer-model gpt-4o-mini --pi-provider openai \
  --methods reward_shaping,generalization \
  --baseline-dir runs/airline-baseline-001 \
  --output-dir runs/airline-pi-001
```

If Pi is not installed, use `--optimizer llm` and omit `--pi-provider` to test
the same phases with the one-shot LLM optimizer. That is **not** a Pi run.
Use a new `--output-dir` on every run, and keep the split, model, trial count,
step limit, and tool source unchanged when reusing a baseline.

Open `runs/airline-pi-001/report.html` in a browser and inspect `summary.json`:
`candidate_applied` means an edit passed validation and both optimized phases
ran; `no_edit` or `optimizer_failed` does not demonstrate candidate efficacy.
The phase JSON files contain raw trajectories and task rewards; optimizer cost
is reported separately from agent-session cost. Local Metaflow also stores the
run artifacts in `.metaflow/` under the directory where you invoked the flow.
It does not provide a hosted UI or a Metaflow Card by default. From that same
directory, you can inspect the latest successful Metaflow run:

```bash
python -c 'from metaflow import Flow; run = Flow("ToolOptimizationHarness").latest_successful_run; print(run.id, run.data.summary["optimization"])'
```

## Run the benchmark adapters without Metaflow

These CLIs are alternatives to the Metaflow run above, not subsequent steps.
Running `ato-tau2` starts a new paid baseline and optimized evaluation; skip
it if you only wanted the Metaflow experiment.

```bash
ato-tau2 \
  --domain airline --split examples/airline-smoke-split.json \
  --agent-model gpt-4o-mini --user-model gpt-4o-mini \
  --optimizer pi --optimizer-model gpt-4o-mini --pi-provider openai \
  --methods reward_shaping,generalization --max-steps 8 --out runs/airline-cli-001
```

The terminal command below is a template, not a copy-paste runnable example:
`splits/tb2.json` is not included, and the model IDs and bundle paths are
placeholders. Create a task-name train/test split and build the Linux OpenCode
source bundle as described in [benchmark setup](docs/benchmarks.md), then
replace those values before running it.

```bash
ato-terminal \
  --benchmark tb2 --benchmark-checkout ../terminal-bench-2 \
  --opencode-checkout ../opencode --split splits/tb2.json \
  --source-bundle ../opencode-source-linux.tar \
  --bun-linux-binary /absolute/path/to/linux/bun --jobs-dir runs/harbor \
  --agent-model provider/agent-model --optimizer-model provider/optimizer-model \
  --methods reward_shaping,generalization --out runs/tb2-001
```

For TBLite, switch to `--benchmark tblite --benchmark-checkout ../OpenThoughts-TBLite`. The CLI runners use `--baseline-only` (Metaflow uses `--skip-optimize`) to create a reusable baseline, then `--baseline-dir <prior-run> --num-candidates 3` for several independent edits. Terminal runs can add `--parallel-phases --n-concurrent 4`. Each run writes phase JSON files, `summary.json`, and `report.html` under the new `--out` directory.

These commands are installed by `python -m pip install -e .`. The module forms (`python -m agent_tool_opt_core.adapters.run_tau2` and `python -m agent_tool_opt_core.adapters.run_terminal`) remain available when a console script is not on `PATH`.

## LLM configuration

The default client delegates provider routing and authentication to [LiteLLM](https://docs.litellm.ai/). Configure providers using LiteLLM's supported configuration mechanisms and use provider-qualified model names where required. This project does not load environment files, manage API keys, or define project IDs. Pi is a separate CLI with its own provider configuration; pass `--pi-provider` when its model requires one.

```python
from agent_tool_opt_core.llm_client import LiteLLMClient

client = LiteLLMClient()
completion = client.complete(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize this tool contract."}],
)

print(completion.text)
print(completion.usage)
print(completion.cost_usd, completion.provider, completion.model)
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for copyright attribution.
