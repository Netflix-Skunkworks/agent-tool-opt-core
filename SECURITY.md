# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Until a public security contact is established, report vulnerabilities privately to the project maintainers through the hosting platform's private vulnerability-reporting feature.

Include the affected version, reproduction steps, expected impact, and any suggested mitigation. Do not include real API keys, private model transcripts, or personal data.

## Credential handling

Provider credentials must be supplied through runtime environment variables or the deployment platform's secret manager. They must never be committed, logged, serialized into experiment artifacts, or passed through command-line arguments.
