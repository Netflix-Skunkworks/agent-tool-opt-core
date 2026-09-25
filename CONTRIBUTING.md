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

Do not commit credentials, private prompts, production traces, personal data, provider headers, or internal infrastructure identifiers. Use synthetic fixtures in tests.

By contributing, you agree that your contribution may be distributed under the repository's approved license once that license is finalized.
