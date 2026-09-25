# Public migration status

This repository is a clean-room migration from a private implementation. Internal Git history and deployment configuration are intentionally excluded.

## Completed

- initialized a clean Git repository;
- migrated the provider-neutral core package and tests;
- replaced in-process internal SDK calls with an injectable LiteLLM client;
- delegated provider routing and authentication to LiteLLM;
- retained normalized usage, cost, and provider provenance;
- declared upstream Metaflow as an optional dependency; and
- added public contribution, security, and CI baselines.

## Pending

- obtain OSPO/legal approval and add the chosen `LICENSE`;
- migrate and sanitize benchmark adapters individually;
- migrate the generic Metaflow harness using upstream Metaflow only;
- remove or rewrite private deployment examples and generated optimizer-I/O captures;
- add public maintainer, support, governance, and release policies;
- run secret, dependency-license, and source-reference audits; and
- validate wheel installation in a clean public-only environment.
