# Security Policy

## Reporting a vulnerability

Report suspected security vulnerabilities privately through
[Netflix's HackerOne program](https://hackerone.com/netflix/), as directed by
[Netflix's Responsible Vulnerability Disclosure policy](https://help.netflix.com/en/node/6657).
Follow the program's current scope and rules; this repository does not expand
testing authorization or promise bounty eligibility. Do not disclose suspected
vulnerabilities in public issues or pull requests.

Include the repository URL, affected version or commit, reproduction steps
using synthetic data, expected impact, and any suggested mitigation. Do not
include real API keys, private model transcripts, or personal data.

## Credential handling

Provider credentials must be supplied through runtime environment variables or
the deployment platform's secret manager. They must never be committed, logged,
serialized into experiment artifacts, or passed through command-line arguments.

## Sharing experiment artifacts

Experiment outputs can contain raw task transcripts, model responses, tool
source, subprocess diagnostics, and local paths. Treat them as sensitive until
reviewed. Inspect and sanitize logs, reports, and attachments before sharing
them in issues, pull requests, or releases. Reproduce problems with synthetic
inputs whenever possible.

## Pi sandbox

An optional [Bubblewrap mode](docs/pi_sandbox.md) restricts the Pi subprocess
on Linux. It shares the host network and does not cover candidate validation
or benchmark execution. Use an isolated worker when running generated code;
the setup guide describes the exact filesystem and credential boundaries.
