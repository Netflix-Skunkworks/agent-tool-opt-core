# Contributing

## Development workflow

1. Create a focused branch from the latest `main`.
2. Add or update tests with every behavioral change.
3. Run formatting, lint, and tests locally.
4. Open a pull request describing the motivation, behavior, validation, and compatibility impact.
5. Resolve review feedback and merge only after all required checks pass.

## Required checks

```bash
ruff check .
ruff format --check .
pytest
```

## Examples and test fixtures

Use synthetic prompts, transcripts, tool results, and diagnostics in tests and
examples. Construct small fixtures that demonstrate the behavior instead of
copying production runs. Use obvious test-only credential placeholders and
generic task IDs, paths, and names.

Do not commit credentials, private prompts, production traces, personal data,
provider headers, or internal infrastructure identifiers. References to public
benchmark task IDs are allowed; document their upstream source and keep the
benchmark data in its separately installed checkout.

Review generated logs, experiment outputs, and source bundles before sharing
them. Follow [SECURITY.md](SECURITY.md) for handling artifacts and reporting
suspected vulnerabilities.

By contributing, you agree that your contribution may be distributed under the repository's Apache-2.0 license.
