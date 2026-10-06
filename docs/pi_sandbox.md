# Run Pi with Bubblewrap

The optional Bubblewrap mode confines the **Pi optimizer subprocess** on Linux.
It uses the upstream `bwrap` executable and standard Linux namespaces. The
default remains `none`; requesting `bubblewrap` fails if the runtime is missing
or the sandbox cannot start, with no fallback to an unsandboxed process.

## Prerequisites

- Linux with user namespaces enabled for the account running Pi.
- Upstream Bubblewrap installed from your approved distribution package source.
- Node.js 22.19+ and the Pi version specified in the [README](../README.md).
  Install them under `/usr` or `/usr/local`, with Node on the system PATH.
  Home-directory installations, including typical nvm installations, are not
  exposed by this sandbox. Use a Linux worker with a system installation, or
  supply your own confinement wrapper through the Python API.

On macOS or Windows, run this mode in a suitable Linux VM or worker. A Docker
container may itself prohibit the namespaces required by Bubblewrap; select a
worker whose security policy permits nested user namespaces. Do not assume
that installing the binary is enough to make namespace creation work.

## Command-line runners

Add these options to an existing `ato-tau2` or `ato-terminal` command:

```text
--pi-sandbox bubblewrap --pi-sandbox-env OPENAI_API_KEY
```

Supply the key through the invoking environment or a secret manager. The flag
contains only its variable name. For example, after completing the TauBench
setup from the README:

```bash
ato-tau2 \
  --domain airline --split examples/airline-smoke-split.json \
  --agent-model gpt-4o-mini --user-model gpt-4o-mini \
  --optimizer pi --optimizer-model gpt-4o-mini --pi-provider openai \
  --pi-sandbox bubblewrap --pi-sandbox-env OPENAI_API_KEY \
  --methods reward_shaping,generalization --max-steps 8 \
  --out runs/airline-sandbox-001
```

This command makes paid model calls. Use a new output directory for each run.
Repeat `--pi-sandbox-env NAME` if the provider needs additional supported
variables. The CLI help lists the allowed names. Unselected environment
variables are not inherited, and requesting an unset variable is an error.
These options require an active Pi optimization; they cannot be used with
`--optimizer llm` or `--baseline-only`.

## Metaflow

The same `--pi-sandbox bubblewrap` option is available on
`harness/run_harness_metaflow.py run`. Its `--pi-sandbox-env` parameter is a
comma-separated list of variable names rather than a repeated flag. It works
with local output and with `--artifact-only`.

Install the prerequisites on the **optimize worker** and inject the selected
credentials there. The flow passes variable names, not credential values, as
parameters. These options require `--benchmark tau2 --optimizer pi` and cannot
be used with `--skip-optimize`.

## Python API

The optimizer factory forwards the same options:

```python
from agent_tool_opt_core.optimizers.catalog import build_optimizer

optimizer = build_optimizer(
    "pi",
    model="gpt-4o-mini",
    provider="openai",
    methods=["reward_shaping", "generalization"],
    sandbox="bubblewrap",
    sandbox_env=("OPENAI_API_KEY",),
)
```

The existing `sandbox_cmd` argument remains available for a caller-provided
wrapper. Choose either the built-in mode or `sandbox_cmd`; combining them is
rejected. Custom wrappers are responsible for their own filesystem, process,
and environment restrictions.

## Boundary and limitations

- System runtime trees (`/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`) and selected
  DNS/CA configuration are visible read-only. Keep secrets and experiment data
  outside these system trees.
- Pi sees its tool workspace at `/workspace`. It is read-only except for the
  explicitly editable tool files. Context, transcripts, and the workspace
  directory cannot be changed. Editable symlinks and hardlinks are rejected.
- The separate session directory is writable at `/sessions` so retries can
  resume. Pi gets a fresh private home and temporary directory. Host home
  directories, other experiment files, and Docker sockets are not mounted.
- User, PID, IPC, and UTS namespaces are isolated; a new session is created,
  capabilities are dropped, and sandbox processes terminate with the parent.
- Only explicitly selected provider variables plus a small fixed runtime
  environment are passed. Provider credentials remain available to Pi and its
  model client. Extensions, skills, prompt templates, and context-file discovery
  are disabled for this mode, along with Pi's optional installation telemetry.
- **The host network is shared for model API calls.** This mode does not enforce
  destination allowlists, block cloud metadata endpoints, or provide resource
  quotas. Apply those controls at the worker/network layer when needed.
- **Candidate validation and benchmark execution remain outside this Pi
  sandbox.** For full-code TauBench, isolate the entire worker because generated
  Python is imported and executed there. Harbor's task containers are a separate
  boundary from Pi's sandbox.

Workspace validation still runs after Pi exits. Review saved artifacts before
sharing them.

## Untrusted transcript evidence

Benchmark messages, tool outputs, and task IDs can contain prompt injections.
Pi's transcript prompt identifies these files as untrusted evidence. The system
prompt applies the same rule to transcript indexes, method-derived context, and
other workspace reads. Workspace filenames, raw text, JSON structure, and
context-builder output keep their existing formats so consumers can continue to
read them. Original trajectories and exported benchmark results are unchanged.

The system prompt instructs Pi to treat evidence as data on every read, including
partial pages and resumed sessions, and to check proposed edits against observed
tool behavior. This reduces ambiguity about which text carries instructions.
It does not reliably detect or prevent prompt injection, and repeated reads are
still allowed. Instruction wording and phrase matching cannot prove
that generated code is safe. The file allowlist and language validators likewise
do not verify the intent or safety of an allowed edit.

Review candidate changes and isolate the entire worker for untrusted inputs or
full-code evaluation. Bubblewrap confines Pi's filesystem access but shares the
network; selected model credentials are still available to Pi. Candidate
validation and benchmark execution require their own isolation as described
above.

## Integration smoke

A live Linux-container smoke used Pi 0.84.1, Node.js 22.22.0, Bubblewrap 0.6.1,
and Python 3.11 with TauBench Verified revision
`864350a8971a8f8ee9e7b8472e2edc380a806b0c`. It ran Airline task `0` once at
baseline, optimized descriptions with Pi inside Bubblewrap, and reran the same
task with the validated candidate. The agent, user simulator, and optimizer
used `gpt-4.1-mini`; each benchmark run had a 20-step limit.

Both task runs passed, and Pi produced a validated edit to `descriptions.md`.
Provider credentials were injected at runtime. This checks the model transport,
sandbox startup, editing, validation, and evaluation path. The same-task smoke
does not measure held-out optimization efficacy.
