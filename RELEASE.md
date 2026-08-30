# Release Guide

This document covers the full process for cutting a numpy-ts release. Steps must be done in order; don't skip ahead.

Requires: conda (`conda activate py313`), npm publish access, and write access to the repo.

---

## Docs URL layout

Read this before touching `docs/`. The layout exists for SEO reasons and the release steps below only make sense against it.

| Path | Role | Indexed? |
|-|-|-|
| `docs/latest/` | The live docs. Serves `https://numpyts.dev/latest/...` and never gets renamed. | Yes — this is the canonical copy |
| `docs/v1.6.x/`, `docs/v1.5.x/`, ... | Frozen snapshots of shipped versions | Only via `canonical` pointing forward to `latest`, or `noindex` where the page no longer exists |
| `docs/next/` | Unreleased edits for the cycle in progress | No — every page carries `noindex: true` |

The stable URL is the point. Every page in `latest/` keeps its URL across releases, so backlinks and accumulated search ranking survive a version bump instead of resetting onto a freshly minted `/v1.8.x/...` path. Archived snapshots are the copies that get new URLs, and a brand-new URL has no ranking to lose.

Two consequences worth internalizing:

- The `version` label in `docs.json` is dropdown text only — it never appears in a URL. `"version": "v1.7.x"` pointing at pages under `latest/` is correct and intended.
- The `redirects` array points at `/latest/...` and should never need editing again. If you find yourself rewriting it during a release, something has gone wrong.

---

## 1. Prep

1. Decide the new version number following semver (e.g. `1.8.0`).
2. Bump version in `package.json`.
3. Draft the changelog entry in `docs/changelog.mdx` (add a new `<Update>` block at the top). Link to `/latest/...`, never to `/next/...` or a version folder.

---

## 2. Full Test Suite

Run the complete test suite across all runtimes. All must pass before continuing.

```bash
# Core suite (lint + typecheck + all test projects)
pnpm run test:ci

# Additional runtimes
pnpm run test:bun
pnpm run test:deno
```

If any test fails, fix it before moving on. Do not skip.

---

## 3. Benchmarks

Run the full benchmark suite across all configurations. This takes a while — plan accordingly.

```bash
pnpm run bench:all
```

This runs Node.js (standard, small, large), Pyodide, and other runtime comparisons and writes results to `benchmarks/results/`.

Sanity-check the results: compare headline numbers to the previous release (`pnpm run bench:compare`) and flag any unexpected regressions before publishing.

---

## 4. Docs

### 4a. Archive the outgoing version

`docs/latest/` currently holds the version you are replacing. Snapshot it under that version's number before overwriting:

```bash
cp -R docs/latest docs/v1.7.x   # the version currently live, not the one you are cutting
```

### 4b. Canonicalize the snapshot

Point every archived page at its live equivalent so the snapshot doesn't compete with `latest/` in search results:

```bash
python3 scripts/archive-docs-version.py archive v1.7.x
```

This adds `canonical: https://numpyts.dev/latest/<path>` to each page that still exists in `docs/latest/`, and `noindex: true` to each page that does not. Never both on one page: `noindex` is a directive Google obeys and `canonical` is a hint it may ignore, so a page carrying both leaves the index without forwarding anything to the live copy. A canonical pointing at a URL that 404s is worse than no canonical, which is why the split is driven by whether the counterpart actually exists.

### 4c. Promote `next/` into `latest/`

```bash
rm -rf docs/latest && cp -R docs/next docs/latest
```

Then strip the `noindex: true` frontmatter that `docs/next/` pages carry — `latest/` must be indexable:

```bash
python3 scripts/archive-docs-version.py unset-noindex latest
```

### 4d. Update `docs.json`

Only one edit per release now:

1. Add a nav block for the version you just archived. Don't hand-edit 85 page paths — generate it:

```bash
python3 scripts/archive-docs-version.py emit-nav v1.7.x --insert
```

2. Update the *label* on the first block to the new version (`"version": "v1.8.x"`). Its page paths stay under `latest/` and keep `"tag": "Latest"` — do not touch them.
3. Leave `redirects` alone. They point at `/latest/...` and are version-independent by design.

### 4e. Benchmark → Docs

```bash
pnpm run bench:docs
```

This runs `scripts/generate-bench-docs.py` and `scripts/generate-overview-charts.py`. Pages are written to `docs/latest/performance/`; the JSON and chart assets they reference are written to `docs/assets/<version label>/` so an archived snapshot keeps pointing at the numbers it shipped with. Review the generated MDX diffs before committing.

Do not skip this on a release where performance changed. Skipping it is why `v1.6.x` shipped serving `v1.5.x` benchmark data.

### 4f. Reseed `docs/next/` for the next cycle

```bash
rm -rf docs/next && cp -R docs/latest docs/next
python3 scripts/archive-docs-version.py set-noindex next
```

`docs/next/` must stay `noindex` — it is a byte-identical copy of `latest/` and would otherwise be crawled as duplicate content.

### 4g. Verify locally

```bash
mintlify dev
mintlify broken-links
```

Check that the new version renders, the version dropdown lists the archived version, and `/quickstart` still lands on `/latest/guides/quickstart`.

Then run the automated check, which covers marker placement, canonical targets, asset references, absolute links and navigation paths:

```bash
pnpm run docs:check
```

It also runs as part of `pnpm run test:ci`, so a broken release leaves CI red rather than shipping quietly. Finally, spot-check the rendered output, which is the part no static check can confirm:

- View source on a `latest/` page — the canonical should be self-referential (`https://numpyts.dev/latest/...`).
- View source on an archived page — the canonical should point forward into `latest/`.
- View source on a `next/` page — it should carry `noindex`.

---

## 5. Merge and Release

Open a PR from the release branch into `main`. Once approved:

1. Merge the PR — this triggers Mintlify auto-publish and the new docs version goes live immediately.
2. Tag the merge commit and push:

```bash
git tag v1.x.x   # e.g. v1.8.0
git push origin v1.x.x
```

Pushing the tag triggers the npm release. Confirm the published package at `https://www.npmjs.com/package/numpy-ts`.

---

## Hotfix / Patch Releases

For patch releases (bug fixes only, no new API surface):

- Skip the benchmarks step unless performance-sensitive code changed.
- Doc changes go directly into `docs/latest/` — there is no version folder to promote and no archiving step, since the version number in the dropdown label does not change for a patch.
- Update `docs/changelog.mdx` with a new `<Update>` block.
- Run `pnpm run test:ci`
