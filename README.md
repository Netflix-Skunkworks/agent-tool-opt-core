# Agent Tool Optimization Core

Provider-neutral building blocks for improving the tools exposed to AI agents from execution transcripts and task rewards.

> **Status:** pre-release migration. The public API and packaging may change before the first stable release.

## Features

- common interfaces for benchmarks, agents, tool targets, validators, and optimizers;
- Pi, one-shot LLM, DRAFT, GEPA, and ToolObserver optimization strategies;
- composable reward-shaping and generalization methods;
- train/test separation and paired evaluation utilities;
- normalized provider usage, estimated cost, and provenance metadata; and
- an optional dependency on upstream Metaflow; the public orchestration harness is still being migrated.

## Installation

The package has not been released to PyPI. Install it from a checkout of this repository:

```bash
git clone https://github.com/Netflix-Skunkworks/agent-tool-opt-core.git
cd agent-tool-opt-core
python -m pip install -e .
```

The `metaflow` extra installs upstream Metaflow. The public orchestration harness is still being migrated.

To smoke-test the public phase runner under Metaflow's local backend without a model key or Titus:

```bash
python -m pip install -e ".[metaflow]"
python examples/metaflow_local_smoke.py show
python examples/metaflow_local_smoke.py run --output-dir runs/metaflow-smoke
```

This synthetic example verifies local Metaflow execution and persisted phase artifacts; it is not the full benchmark Metaflow harness.

## Benchmark source checkouts

The benchmark datasets and agent code are not bundled with this package. To run tool optimization against TauBench Verified, TerminalBench 2, or OpenThoughts TBLite, clone their public repositories next to `agent-tool-opt-core`:

```bash
git clone https://github.com/amazon-agi/tau2-bench-verified.git ../tau2-bench-verified
git clone https://github.com/laude-institute/terminal-bench-2.git ../terminal-bench-2
git clone https://github.com/open-thoughts/OpenThoughts-TBLite.git ../OpenThoughts-TBLite
git clone https://github.com/anomalyco/opencode.git ../opencode
git clone https://github.com/laude-institute/harbor.git ../harbor
```

The TauBench adapter imports the local `tau2` package. The TerminalBench 2 and TBLite adapters run tasks from their local checkouts through Harbor, with OpenCode source and edited tool files uploaded into each sandbox. Both targets support description-only edits by default and opt-in full-code edits. The public runners preserve the five-phase baseline-train/test → optimize → optimized-train/test setup, with baseline reuse, multiple candidates, bounded parallel terminal phases, and an HTML summary. See [benchmark setup](docs/benchmarks.md) for installation, frozen task splits, and runnable commands. Terminal runs require a Linux OpenCode dependency bundle and Linux Bun executable; these are not shipped here. Non-billable local adapter smokes passed; real-model benchmark completion still needs a valid provider key.

## Run the benchmark adapters

First install the cloned TauBench and Harbor packages (Harbor requires Python 3.12+), create JSON files with disjoint `train` and `test` task IDs, and configure credentials for your chosen models:

```bash
python -m pip install -e ../tau2-bench-verified
python -m pip install -e ../harbor
```

The terminal adapter also needs the Linux OpenCode source bundle described in [benchmark setup](docs/benchmarks.md). Replace the model names and paths below with your own:

```bash
ato-tau2 \
  --domain airline --split splits/airline.json \
  --agent-model provider/agent-model --user-model provider/user-model \
  --optimizer pi --optimizer-model provider/optimizer-model \
  --methods reward_shaping,generalization --out runs/airline-001

ato-terminal \
  --benchmark tb2 --benchmark-checkout ../terminal-bench-2 \
  --opencode-checkout ../opencode --split splits/tb2.json \
  --source-bundle ../opencode-source-linux.tar \
  --bun-linux-binary /absolute/path/to/linux/bun --jobs-dir runs/harbor \
  --agent-model provider/agent-model --optimizer-model provider/optimizer-model \
  --methods reward_shaping,generalization --out runs/tb2-001
```

For TBLite, switch to `--benchmark tblite --benchmark-checkout ../OpenThoughts-TBLite`. Use `--baseline-only` to create a reusable baseline, then `--baseline-dir <prior-run> --num-candidates 3` for several independent edits. Terminal runs can add `--parallel-phases --n-concurrent 4`. Each run writes phase JSON files, `summary.json`, and `report.html` under the new `--out` directory.

These commands are installed by `python -m pip install -e .`. The module forms (`python -m agent_tool_opt_core.adapters.run_tau2` and `python -m agent_tool_opt_core.adapters.run_terminal`) remain available when a console script is not on `PATH`.

## LLM configuration

The default client delegates provider routing and authentication to [LiteLLM](https://docs.litellm.ai/). Configure providers using LiteLLM's supported configuration mechanisms and use provider-qualified model names. This project does not load environment files, manage API keys, or define project IDs.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before opening a pull request or reporting a vulnerability.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for copyright attribution.
