# Metaflow artifacts for remote execution

The TauBench flow does not choose or deploy a remote executor. Configure your
Metaflow metadata service, durable datastore, and execution backend first, then
run the flow with `--artifact-only` under that backend. Without a remote
execution option, these commands still execute on the local machine.

## Worker environment

Each step needs a compatible, pinned worker environment containing this repo,
upstream TauBench Verified and its task data, and the `.[metaflow]` Python
dependencies. The optimize step also needs Node.js 22.19+ and a compatible
`pi` CLI (tested with `@earendil-works/pi-coding-agent@0.84.1`) when using
`--optimizer pi`. A prebuilt worker image is generally simpler than
installing dependencies separately in each step. Provide model credentials
through the backend's secret mechanism; never put key values in flow parameters,
source bundles, or checked-in files.

In artifact-only mode, the Pi subprocess gets a fresh home directory and a
limited environment: runtime path/certificate settings plus OpenAI, Anthropic,
or Google API-key variables when present. Backend metadata credentials are not
forwarded as environment variables. This is not a filesystem or network
sandbox: isolate the optimize worker and restrict its workload identity and
mounted files. Raw task transcripts and optimizer events are persisted as
Metaflow artifacts, so restrict datastore access and retention accordingly.

## Inputs and outputs

`--split` names a JSON file on the submitting machine. Metaflow `IncludeFile`
packages the contents with the run; workers do not need the original path.
With `--artifact-only`, do not pass `--output-dir` or `--baseline-dir`: steps
use worker-local scratch space and persist outputs in the Metaflow datastore.
The run exposes `train_phase`, `test_phase`, `optimized_artifacts`, `candidates`,
`optimizer_artifacts`, `summary`, and `report_html`. Optimizer audit files are
captured up to 10 MB per file.

Apply your configured backend's normal execution or deployment options to
these commands. The first run records a reusable baseline:

```bash
python harness/run_harness_metaflow.py run \
  --artifact-only --benchmark tau2 --domain airline \
  --split examples/airline-smoke-split.json \
  --agent-model gpt-4o-mini --user-model gpt-4o-mini \
  --num-trials 1 --max-steps 8 --skip-optimize
```

Record its Metaflow run ID. A second run can reuse its paired baseline without
accessing files from the first worker:

```bash
python harness/run_harness_metaflow.py run \
  --artifact-only --benchmark tau2 --domain airline \
  --split examples/airline-smoke-split.json \
  --agent-model gpt-4o-mini --user-model gpt-4o-mini \
  --num-trials 1 --max-steps 8 \
  --optimizer pi --optimizer-model gpt-4o-mini --pi-provider openai \
  --methods reward_shaping,generalization \
  --baseline-run-id BASELINE_RUN_ID
```

The source run must be complete and reachable through the same Metaflow
metadata service and namespace. The harness checks benchmark, agent, task IDs,
trial settings, and tool snapshot before reusing its train/test baselines.

Read the results from an environment configured for that datastore:

```python
from pathlib import Path

from metaflow import Run

run = Run("ToolOptimizationHarness/RUN_ID")
print(run.data.summary)
print(run.data.optimized_artifacts)
Path(f"report-{run.id}.html").write_text(run.data.report_html, encoding="utf-8")
```

Open the exported HTML file in a browser. The flow does not create a Metaflow
Card automatically.

The artifact-only path and run-ID baseline reuse are exercised in CI under
local Metaflow, which runs steps in separate processes. No remote backend is
claimed verified until it passes a backend-specific synthetic run and a small
real-model A/B run.
