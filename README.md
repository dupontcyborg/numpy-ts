# numpy-ts

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![npm version](https://img.shields.io/npm/v/numpy-ts)](https://www.npmjs.com/package/numpy-ts)
![bundle size](https://img.shields.io/bundlejs/size/numpy-ts)
![numpy api coverage](https://img.shields.io/badge/numpy_api_coverage-94%20%25-brightgreen)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero-dark.svg" />
  <img src="docs/images/hero-light.svg" alt="numpy-ts" />
</picture>

Complete NumPy implementation for TypeScript and JavaScript.

- **📊 Extensive API** — **476 of 507 NumPy functions (93.9% coverage)**
- **✅ NumPy-validated** — 6,000+ tests compared against Python NumPy
- **🔒 Type-safe** — Full TypeScript type definitions
- **🌳 Tree-shakeable** — Import only what you use
- **🌐 Universal** — Works in Node.js and browsers

[Docs](https://https://numpy-ts.dev) • [Examples](https://numpy-ts.dev/examples) • [Benchmarks](https://numpy-ts.dev/performance)

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

- Docs: https://numpy-ts.dev
- Playground: https://numpy-ts.dev/playground
- Usage Examples: https://numpy-ts.dev/examples
- API Coverage Report: https://numpy-ts.dev/coverage
- Performance Benchmarks: https://numpy-ts.dev/performance

## Contributing

Issues and PRs are welcome: https://github.com/dupontcyborg/numpy-ts

## License

MIT © Nicolas Dupont
