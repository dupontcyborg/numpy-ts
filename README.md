# numpy-ts

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![npm version](https://img.shields.io/npm/v/numpy-ts)](https://www.npmjs.com/package/numpy-ts)
![bundle size](https://img.shields.io/bundlejs/size/numpy-ts)
![numpy api coverage](https://img.shields.io/badge/numpy_api_coverage-94%20%25-brightgreen)

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
- **✅ NumPy-validated** — 8,000+ tests compared against Python NumPy
- **🔒 Type-safe** — Full TypeScript type definitions
- **🌳 Tree-shakeable** — Import only what you use (`np.add()` -> ~10kB bundle)
- **🌐 Universal** — Works in Node.js, Deno, Bun and browsers
- **🏃🏽‍♂️ Fast** - Zig WASM kernels competitive with Python NumPy (see [benchmarks](https://numpyts.dev/performance))

[Docs](https://numpyts.dev) • [Examples](https://numpyts.dev/examples) • [Benchmarks](https://numpyts.dev/performance)

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

## Resources

- Docs: https://numpyts.dev
- Playground: https://numpyts.dev/playground
- Usage Examples: https://numpyts.dev/examples
- API Coverage Report: https://numpyts.dev/coverage
- Performance Benchmarks: https://numpyts.dev/performance

## Contributing

Issues and PRs are welcome: https://github.com/dupontcyborg/numpy-ts

## License

MIT © Nicolas Dupont
