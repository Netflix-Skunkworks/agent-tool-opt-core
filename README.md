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

## Benchmark source checkouts

The benchmark datasets and agent code are not bundled with this package. To run tool optimization against TauBench Verified, TerminalBench 2, or OpenThoughts TBLite, clone their public repositories next to `agent-tool-opt-core`:

```bash
git clone https://github.com/amazon-agi/tau2-bench-verified.git ../tau2-bench-verified
git clone https://github.com/laude-institute/terminal-bench-2.git ../terminal-bench-2
git clone https://github.com/open-thoughts/OpenThoughts-TBLite.git ../OpenThoughts-TBLite
git clone https://github.com/anomalyco/opencode.git ../opencode
git clone https://github.com/laude-institute/harbor.git ../harbor
```

The TauBench adapter imports the local `tau2` package. The TerminalBench 2 and TBLite adapters run tasks from their local checkouts through Harbor, with OpenCode source and edited tool descriptions uploaded into each sandbox. See [benchmark setup](docs/benchmarks.md) for installation, frozen task splits, and example adapter construction. Terminal runs require a Linux OpenCode dependency bundle and Linux Bun executable; these are not shipped here. Live benchmark runs have not yet been verified in this public migration.

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

License selection and approval are pending. Do not publish a release until an approved `LICENSE` file is added.
