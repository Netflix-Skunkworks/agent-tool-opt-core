# Agent Tool Optimization Core

Provider-neutral building blocks for improving the tools exposed to AI agents from execution transcripts and task rewards.

> **Status:** pre-release migration. The public API and packaging may change before the first stable release.

## Features

- common interfaces for benchmarks, agents, tool targets, validators, and optimizers;
- Pi, one-shot LLM, DRAFT, GEPA, and ToolObserver optimization strategies;
- composable reward-shaping and generalization methods;
- train/test separation and paired evaluation utilities;
- normalized provider usage, estimated cost, and provenance metadata; and
- optional upstream Metaflow orchestration.

## Installation

```bash
pip install agent-tool-opt-core
```

Metaflow support is optional:

```bash
pip install "agent-tool-opt-core[metaflow]"
```

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
