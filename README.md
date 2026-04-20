# numpy-ts

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![npm version](https://img.shields.io/npm/v/numpy-ts)](https://www.npmjs.com/package/numpy-ts)
![npm package minimized gzipped size](https://img.shields.io/bundlejs/size/numpy-ts?label=size%20(full))
![npm package minimized gzipped size](https://img.shields.io/bundlejs/size/numpy-ts?exports=wasmConfig&label=size%20(single))
![numpy api coverage](https://img.shields.io/badge/numpy_api_coverage-94%20%25-brightgreen)
[![Sponsored by Cyborg](https://img.shields.io/badge/sponsored_by-Cyborg-35A8B1?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgNTEyIDUxMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBmaWxsPSIjZmZmIiBkPSJNMjYwLjggMTEwLjlDMjMzLjYgMTEwLjkgMjA2LjkgMTE4LjUgMTgzLjggMTMyLjlDMTYwLjcgMTQ3LjMgMTQyLjEgMTY3LjkgMTMwLjEgMTkyLjNIMjMzLjhDMjQyLjMgMTg4LjggMjUxLjQgMTg2LjkgMjYwLjYgMTg2LjlDMjY5LjggMTg2LjkgMjc4LjkgMTg4LjcgMjg3LjQgMTkyLjJDMjk1LjkgMTk1LjggMzAzLjcgMjAxIDMxMC4xIDIwNy41QzMxNi42IDIxNC4xIDMyMS43IDIyMS44IDMyNS4yIDIzMC40SDQwMy44QzM5Ny43IDE5Ni45IDM4MC4xIDE2Ni42IDM1NCAxNDQuN0MzMjcuOSAxMjIuOSAyOTQuOSAxMTAuOSAyNjAuOSAxMTAuOUgyNjAuOFoiLz48cGF0aCBmaWxsPSIjZmZmIiBkPSJNMTIxLjkgMjk5LjhDMTMxLjIgMzI5LjQgMTQ5LjcgMzU1LjIgMTc0LjcgMzczLjVDMTk5LjYgMzkxLjggMjI5LjggNDAxLjcgMjYwLjggNDAxLjdDMjkxLjcgNDAxLjcgMzIxLjkgMzkxLjggMzQ2LjggMzczLjVDMzcxLjggMzU1LjIgMzkwLjMgMzI5LjQgMzk5LjYgMjk5LjhIMzE0LjhDMzA0LjQgMzEyLjggMjg5LjYgMzIxLjUgMjczLjMgMzI0LjVDMjU2LjkgMzI3LjUgMjQwIDMyNC41IDIyNS43IDMxNi4xQzIxMS40IDMwNy43IDIwMC41IDI5NC40IDE5NS4xIDI3OC43QzE4OS43IDI2Mi45IDE5MC4xIDI0NS44IDE5Ni4zIDIzMC40SDc1LjhDODEuMiAxOTEgOTkuMSAxNTQuNCAxMjYuOCAxMjUuOEMxNTQuNSA5Ny4zIDE5MC42IDc4LjQgMjI5LjggNzEuOEMyNjkuMSA2NS4yIDMwOS40IDcxLjMgMzQ0LjkgODkuMkMzODAuNCAxMDcuMSA0MDkuMiAxMzUuOSA0MjcuMyAxNzEuM0g1MDIuM0M0ODEgMTEwLjcgNDM3LjggNjAuMyAzODEuMSAzMC4xQzMyNC40IC0wLjEgMjU4LjUgLTcuOSAxOTYuMyA4LjNDMTM0LjIgMjQuNSA4MC40IDYzLjQgNDUuNiAxMTcuNUMxMC45IDE3MS41IC0yLjIgMjM2LjYgOC44IDI5OS44SDEyMS45WiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Ik00MzQuMiAzMjUuN0M0MjAuNyAzNTkuMyAzOTcuOCAzODguMiAzNjguMiA0MDkuMUMzMzguNiA0MjkuOSAzMDMuNiA0NDEuNiAyNjcuNSA0NDIuOUMyMzEuMyA0NDQuMiAxOTUuNiA0MzUgMTY0LjYgNDE2LjRDMTMzLjUgMzk3LjcgMTA4LjYgMzcwLjUgOTIuOCAzMzhIMTguNEMzNS44IDM4OS43IDY5LjQgNDM0LjUgMTE0LjIgNDY1LjhDMTU5IDQ5Ny4xIDIxMi42IDUxMy4zIDI2Ny4yIDUxMS45QzMyMS44IDUxMC42IDM3NC41IDQ5MS43IDQxNy43IDQ1OC4yQzQ2MC44IDQyNC43IDQ5Mi4xIDM3OC4zIDUwNyAzMjUuN0g0MzQuMloiLz48L3N2Zz4=)](https://www.cyborg.co/?ref=numpyts.dev)

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="https://raw.githubusercontent.com/dupontcyborg/numpy-ts/main/docs/images/hero-dark.svg"
  />
  <img
    src="https://raw.githubusercontent.com/dupontcyborg/numpy-ts/main/docs/images/hero-light.svg"
    alt="numpy-ts"
  />
</picture>

Complete NumPy implementation for TypeScript and JavaScript.

- **📊 Extensive API** — **476 of 507 NumPy functions (93.9% coverage)**
- **🏃🏽‍♂️ Fast** — [1.13x faster than NumPy](https://numpyts.dev/performance) on average across 7,200 benchmarks, thanks to Zig-WASM SIMD kernels
- **✅ NumPy-validated** — 20,000+ tests compared against Python NumPy
- **🔒 Type-safe** — Full TypeScript type definitions
- **🌳 Tree-shakeable** — Import only what you use (`np.add()` -> ~10kB bundle)
- **🌐 Universal** — Zero dependencies, works in Node.js, Deno, Bun and browsers

[Docs](https://numpyts.dev) • [Playground](https://numpyts.dev/playground) • [Examples](https://numpyts.dev/examples) • [Coverage](https://numpyts.dev/coverage) • [Benchmarks](https://numpyts.dev/performance)

## Install

```bash
npm install numpy-ts
```

## Quick Start

```typescript
import * as np from 'numpy-ts';

// Array creation with dtype support
const A = np.array([[1, 2], [3, 4]], 'float32');
const B = np.ones([2, 2], 'int32');

// Broadcasting and chained operations
const result = A.add(5).multiply(2);

// Linear algebra
const C = A.matmul(B);
const trace = A.trace();

// Reductions with axis support
const colMeans = A.mean(0);  // [2.0, 3.0]

// NumPy-style slicing with strings
const row = A.slice('0', ':');    // A[0, :]
const submatrix = A.slice('0:2', '1:');  // A[0:2, 1:]
```

## Sponsors

<p align="center">
  numpy-ts is brought to you by<br><br>
  <a href="https://www.cyborg.co/?ref=numpyts.dev">
    <picture>
      <source
        media="(prefers-color-scheme: dark)"
        srcset="https://raw.githubusercontent.com/dupontcyborg/numpy-ts/main/docs/images/sponsors/cyborg-dark.svg"
      />
      <img
        src="https://raw.githubusercontent.com/dupontcyborg/numpy-ts/main/docs/images/sponsors/cyborg-light.svg"
        alt="Cyborg"
        width="200"
      />
    </picture>
  </a>
</p>

<p align="center">
  <sub>Interested in supporting numpy-ts? <a href="https://github.com/sponsors/dupontcyborg">Become a sponsor</a>.</sub>
</p>

## Contributing

Issues and PRs are welcome: https://github.com/dupontcyborg/numpy-ts

## License

MIT © Nicolas Dupont
