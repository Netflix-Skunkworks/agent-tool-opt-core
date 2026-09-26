# Public benchmark checkouts

The benchmark and agent repositories are separate from `agent-tool-opt-core`. Clone them next to this repository; their code and datasets are not redistributed here.

| Evaluation | Public upstream | Local checkout | Purpose |
|---|---|---|---|
| TauBench Verified | [amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified) | `../tau2-bench-verified` | Airline, retail, and telecom simulations and tools |
| TerminalBench 2 | [laude-institute/terminal-bench-2](https://github.com/laude-institute/terminal-bench-2) | `../terminal-bench-2` | Terminal tasks and verifiers |
| OpenThoughts TBLite | [open-thoughts/OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite) | `../OpenThoughts-TBLite` | Smaller terminal task set |
| OpenCode | [anomalyco/opencode](https://github.com/anomalyco/opencode) | `../opencode` | Agent tool descriptions and implementation for both terminal evaluations |

```bash
git clone https://github.com/amazon-agi/tau2-bench-verified.git ../tau2-bench-verified
git clone https://github.com/laude-institute/terminal-bench-2.git ../terminal-bench-2
git clone https://github.com/open-thoughts/OpenThoughts-TBLite.git ../OpenThoughts-TBLite
git clone https://github.com/anomalyco/opencode.git ../opencode
```

Install TauBench Verified from its checkout in the environment where you run the adapter:

```bash
python -m pip install -e ../tau2-bench-verified
```

The terminal evaluations use [Harbor](https://github.com/laude-institute/harbor). The public TBLite project documents `harbor run --dataset openthoughts-tblite`; TerminalBench 2 documents `harbor run --dataset terminal-bench@2.0`. Keep baseline and candidate evaluations on the same task IDs, agent model, and Harbor environment. When optimizing OpenCode tool text, use an agent run from the local OpenCode source: a prebuilt agent may have the descriptions bundled at build time, leaving edits ineffective.

The benchmark projects change independently. Record each checkout's Git commit alongside results and confirm its own dependency and license terms before distributing any derived artifacts.
