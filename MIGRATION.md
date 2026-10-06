# Implementation status

This document summarizes the functionality available in `agent-tool-opt-core`.

## Implemented

- Provider-neutral core package and tests, with an injectable LiteLLM client
  for provider routing and authentication.
- Normalized usage, cost, and provider provenance.
- TauBench Verified, TerminalBench 2, and TBLite adapters using separately
  installed upstream code and datasets.
- Command-line benchmark runners and an optional upstream Metaflow harness,
  including synthetic smoke execution and datastore artifact support.
- Optional upstream Bubblewrap confinement for the Pi subprocess, exposed by
  the CLI runners, Python API, and Metaflow harness.
- Apache-2.0 [LICENSE](LICENSE) and copyright [NOTICE](NOTICE).
- [Contribution guidelines](CONTRIBUTING.md), a
  [vulnerability-reporting policy](SECURITY.md), and CI for formatting, lint,
  unit tests, and synthetic Metaflow smoke runs.

Remote execution requires backend-specific validation as described in
[the Metaflow guide](docs/metaflow_remote.md). Local CI smoke checks do not
establish that every remote backend has been tested.
