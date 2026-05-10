# Release Guide

This document covers the full process for cutting a numpy-ts release. Steps must be done in order; don't skip ahead.

Requires: conda (`conda activate py313`), npm publish access, and write access to the repo.

---

## 1. Prep

1. Decide the new version number following semver (e.g. `1.4.0`).
2. Bump version in `package.json`.
3. Draft the changelog entry in `docs/changelog.mdx` (add a new `<Update>` block at the top).

---

## 2. Full Test Suite

Run the complete test suite across all runtimes. All must pass before continuing.

```bash
# Core suite (lint + typecheck + all test projects)
npm run test:ci

# Additional runtimes
npm run test:bun
npm run test:deno
```

If any test fails, fix it before moving on. Do not skip.

---

## 3. Benchmarks

Run the full benchmark suite across all configurations. This takes a while — plan accordingly.

```bash
npm run bench:all
```

This runs Node.js (standard, small, large), Pyodide, and other runtime comparisons and writes results to `benchmarks/results/`.

Sanity-check the results: compare headline numbers to the previous release (`npm run bench:compare`) and flag any unexpected regressions before publishing.

---

## 4. Benchmark → Docs

Generate performance documentation from the benchmark results:

```bash
npm run bench:docs
```

This runs `scripts/generate-bench-docs.py` and `scripts/generate-overview-charts.py`, updating the performance pages in `docs/next/performance/`. Review the generated MDX diffs before committing.

---

## 5. Docs Release

### 5a. Promote `next/` to the new version

Rename `docs/next/` to the new version folder:

```bash
mv docs/next docs/v1.x.x   # replace with actual version, e.g. v1.4.x
```

### 5b. Update `docs.json`

In `docs/docs.json`:

1. Add the new version to the `navigation.versions` array (copy the previous version's entry and update all page paths from `v1.3.x/...` to `v1.4.x/...`).
2. Update the top-level `redirects` to point shorthand URLs (e.g. `/quickstart`, `/api`) to the new version.
3. Keep the previous version in the versions list — don't remove it.

### 5c. Seed `docs/next/` for the next cycle

```bash
cp -r docs/v1.x.x docs/next   # copy from the version you just created
```

This ensures `docs/next/` is always a ready-to-edit mirror of the latest version for future contributors.

### 5d. Verify locally

```bash
mintlify dev
mintlify broken-links
```

Check the new version renders correctly and the redirects work before merging.

---

## 6. Merge and Release

Open a PR from the release branch into `main`. Once approved:

1. Merge the PR — this triggers Mintlify auto-publish and the new docs version goes live immediately.
2. Tag the merge commit and push:

```bash
git tag v1.x.x   # e.g. v1.4.0
git push origin v1.x.x
```

Pushing the tag triggers the npm release. Confirm the published package at `https://www.npmjs.com/package/numpy-ts`.

---

## Hotfix / Patch Releases

For patch releases (bug fixes only, no new API surface):

- Skip the benchmarks step unless performance-sensitive code changed.
- Doc changes go directly into the live version folder (`docs/v1.3.x/`) rather than `docs/next/`.
- Update `docs/changelog.mdx` with a new `<Update>` block.
- Run `npm run test:ci`
