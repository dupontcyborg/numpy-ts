# Contributing to numpy-ts

Thanks for your interest in contributing! We welcome contributions of all kinds.

## Setup

```bash
git clone https://github.com/dupontcyborg/numpy-ts.git
cd numpy-ts
npm install
npm test
```

Validation tests and benchmarks require Python + NumPy:
```bash
source ~/.zshrc && conda activate py313
```

## Contribution Checklist

Every new function requires:

1. **Implementation** in `src/` + export in `src/index.ts`
2. **Unit tests** in `tests/unit/` — `npm run test:unit`
3. **Validation tests** in `tests/validation/` — `npm run test:validation` (requires conda)
4. **Benchmarks** in `benchmarks/src/specs.ts`, `runner.ts`, and `scripts/numpy_benchmark.py`
5. **Update docs** in `docs/next/` — update the relevant API reference page and [API coverage](https://numpyts.dev/v1.0.x/guides/api-coverage) (see [Docs Workflow](#docs-workflow) below)
6. **Checks** — `npm run lint && npm test`

## Docs Workflow

The documentation site at [numpyts.dev](https://numpyts.dev) **auto-publishes whenever changes are merged to `main`**. To avoid publishing docs for unreleased features, all contributor doc changes go into `docs/next/` — not into a versioned folder like `docs/v1.3.x/`.

`docs/next/` is a full mirror of the current version. Edit the relevant pages there exactly as you would for the live docs. At release time, `docs/next/` gets renamed to the new version folder (e.g. `docs/v1.4.x/`) and published together with the release.

**Where to make changes:**

| Change type | Where |
|-|-|
| New function docs / API reference | `docs/next/api/` |
| New or updated guide | `docs/next/guides/` |
| New example | `docs/next/examples/` |
| Bug fix in existing published docs | `docs/<version>/` (the live version) |

Bug fixes to existing published docs can go directly into `docs/<version>/` — those changes are safe to publish immediately.

## Submitting

1. Create a branch: `git checkout -b feature/your-feature`
2. Commit, push, and open a PR on GitHub

## Key Resources

- [Documentation site](https://numpyts.dev) — Guides, API reference, and examples
- [API Coverage](https://numpyts.dev/v1.0.x/guides/api-coverage) — Implementation progress

## Questions?

Open an issue at https://github.com/dupontcyborg/numpy-ts/issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
