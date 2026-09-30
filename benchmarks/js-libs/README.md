# numpy-ts vs JS numerical libraries

A **coverage-aware** benchmark comparing numpy-ts against third-party JavaScript
numerical libraries: [@jax-js/jax](https://github.com/ekzhang/jax-js),
[TensorFlow.js](https://github.com/tensorflow/tfjs) (WASM backend), [mathjs](https://mathjs.org),
[numeric.js](https://github.com/sloisel/numeric), [ml-matrix](https://github.com/mljs/matrix),
[@d4c/numjs](https://www.npmjs.com/package/@d4c/numjs), and [@stdlib](https://stdlib.io).

> **These are dev-only deps, isolated in this directory's own `package.json`.**
> numpy-ts itself ships **zero** dependencies — nothing here is a dependency of the
> published package. Root `npm install` does not install any of them.

## Run

```bash
pnpm run bench:js-libs            # float64 regime (mathjs, numeric, ml-matrix, stdlib, numjs)
pnpm run bench:js-libs:f32        # float32 regime (jax-js, tfjs, stdlib, numjs)
# options:
tsx benchmarks/js-libs/run.ts --regime float32 --size small --op add,multiply,matmul
```

First run auto-installs the isolated deps (`bench:js-libs:install`).

## How it works (fairness)

- **Coverage-aware.** Each library's adapter declares only the ops it genuinely
  supports; the runner benchmarks the intersection with numpy-ts's spec list and
  reports per-library coverage (`N / total`). We never imply a library ran ops it
  cannot do. See `COVERAGE.md` for the full matrix + dtype support.
- **Same inputs.** Every library gets byte-identical input values, extracted from
  numpy-ts's own `setupArrays()`.
- **One timer.** `lib/timing.ts` mirrors the project's auto-calibrated methodology
  (warmup → calibrate ops/sample → N samples) and **awaits materialization**, so
  lazy/async libraries (tfjs, jax-js) are forced to fully compute inside the timed
  region — the same logical work the eager libraries do.
- **dtype regimes.** float64 is the cross-library default; float32 is a separate
  regime for the f32-native libs (tfjs has no float64; jax-js is f32-native).
  numpy-ts runs at the regime dtype too — never f32-vs-f64. int64/uint64/float16/
  complex are numpy-ts capability showcases, not races (see COVERAGE.md).
- **Per-library constraints** (jax-js `.ref` ownership + dispose, tfjs functional
  API + `dataSync`, ml-matrix 2D-only, numeric pure-JS) live in each adapter.

## Layout

```
js-libs/
  package.json        # isolated dev deps (own node_modules, gitignored)
  COVERAGE.md         # op + dtype coverage determination
  probe.ts            # dump each library's API surface
  dtype-probe.ts      # dump each library's dtype support
  run.ts              # the runner CLI
  lib/                # types, specdata extraction, async timer, report
  adapters/           # one adapter per library (+ numpy-ts reference)
```

## Adding an operation

Add the op to each library's `ops` map in `adapters/<lib>.ts` (keyed by the
numpy-ts operation name), and add the name to `CORE_OPS` in `run.ts`. The op is
keyed by SpecData (`arrays.a`, `arrays.b`, `params.axis`, …); throw inside
`prepare`/`run` for unsupported param combinations and it's recorded as N/A.
