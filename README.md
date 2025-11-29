# numpy-ts

Complete NumPy implementation for TypeScript and JavaScript

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Under Construction](https://img.shields.io/badge/Under%20Construction-red)

**⚠️ WARNING: Under active development. API may change.**

---

## What is numpy-ts?

A complete NumPy 2.0+ implementation for TypeScript/JavaScript, validated against Python NumPy.

**Goals:**
- **100% API Coverage** - All NumPy functions and operations
- **Type Safety** - Full TypeScript support with inference
- **Correctness** - Cross-validated against Python NumPy
- **Cross-Platform** - Node.js and browsers

**Non-Goals:**
- Matching NumPy's C-optimized performance (initially)
- C API compatibility

---

## Quick Example

```typescript
import * as np from 'numpy-ts';

// Create arrays (default float64 dtype)
const A = np.array([[1, 2], [3, 4]]);
const B = np.zeros([2, 2]);

// Create arrays with specific dtypes (11 types supported)
const intArr = np.ones([3, 3], 'int32');
const floatArr = np.arange(0, 10, 1, 'float32');
const boolArr = np.array([1, 0, 1], 'bool');
const bigIntArr = np.array([1n, 2n, 3n], 'int64');  // BigInt support

// All operations preserve dtype or follow NumPy promotion rules
const result = intArr.add(5);  // Stays int32
const promoted = intArr.add(floatArr);  // Promotes to float32

// Matrix operations
const C = A.matmul(B);
const eigenvalues = np.linalg.eig(A);

// Slicing (string-based syntax)
const row = A.slice('0', ':');  // First row
const col = A.col(1);            // Second column

// Broadcasting (fully implemented!)
const scaled = A.add(5).multiply(2);

// Advanced broadcasting examples:
const row = np.array([1, 2, 3, 4]);        // (4,)
const col = np.array([[1], [2], [3]]);     // (3, 1)
const result = col.multiply(row);           // (3, 4) via broadcasting!

// Reductions
const total = A.sum();                // Sum all elements
const columnMeans = A.mean(0);        // Mean along axis 0
const rowMaxs = A.max(1, true);       // Max along axis 1, keep dims

// Comparisons (return boolean arrays as uint8)
const mask = A.greater(5);            // Element-wise A > 5
const equal = A.equal(B);             // Element-wise A == B
const inRange = A.greater_equal(0);   // A >= 0

// Tolerance comparisons (for floating point)
const close = A.isclose(B);           // Element-wise closeness
const allClose = A.allclose(B);       // True if all elements close

// Reshape operations (view vs copy semantics)
const reshaped = A.reshape(4, 1);     // View if C-contiguous, copy otherwise
const flat = A.flatten();             // Always returns a copy
const ravel = A.ravel();              // View if C-contiguous, copy otherwise
const transposed = A.transpose();      // Always returns a view
const squeezed = A.squeeze();          // Always returns a view
const expanded = A.expand_dims(0);     // Always returns a view

// View tracking (NumPy-compatible)
const view = A.slice('0:2', '0:2');
console.log(view.base === A);         // true - view tracks base array
console.log(view.flags.OWNDATA);      // false - doesn't own data
console.log(A.flags.OWNDATA);         // true - owns data

// Memory layout flags
console.log(A.flags.C_CONTIGUOUS);    // true - C-order (row-major)
console.log(A.flags.F_CONTIGUOUS);    // false - not Fortran-order

// Random
const random = np.random.randn([100, 100]);

// I/O (Node.js)
np.save('matrix.npy', A);
const loaded = np.load('matrix.npy');
```

---

## Architecture

Pure TypeScript implementation with three layers:

```
┌────────────────────────────────┐
│    NumPy-Compatible API        │
└────────────┬───────────────────┘
             │
┌────────────┴───────────────────┐
│  NDArray (Memory & Views)      │
│  Broadcasting, Slicing, DTypes │
└────────────┬───────────────────┘
             │
┌────────────┴───────────────────┐
│  TypeScript/JavaScript Core    │
│  Computational Engine          │
└────────────────────────────────┘
```

Built from scratch for correctness and NumPy compatibility.

---

## Key Features

### Comprehensive NumPy API
- **Array creation**: `zeros`, `ones`, `arange`, `linspace`, `eye` (all support dtype parameter)
- **Arithmetic operations**: `add`, `subtract`, `multiply`, `divide` with broadcasting
- **Linear algebra**: `matmul`, `dot`, `transpose`
- **Reductions**: `sum`, `mean`, `std`, `min`, `max` with axis support
- **DTypes**: 11 types supported (float32/64, int8/16/32/64, uint8/16/32/64, bool)
  - Full dtype preservation across operations
  - NumPy-compatible type promotion
  - BigInt support for int64/uint64
- **NPY/NPZ I/O**: Read and write `.npy` and `.npz` files (v1/v2/v3 compatible) for all supported dtypes
- **View tracking**: `base` attribute tracks view relationships
- **Memory flags**: `C_CONTIGUOUS`, `F_CONTIGUOUS`, `OWNDATA`
- **Comparisons**: `greater`, `less`, `equal`, `isclose`, `allclose`
- **Reshaping**: `reshape`, `flatten`, `ravel`, `transpose`, `squeeze`, `expand_dims`

### TypeScript Native
```typescript
// Full type inference
const arr = np.zeros([3, 4]);  // Type: NDArray<Float64>
arr.shape;  // Type: readonly [3, 4]
arr.sum();  // Type: number

// Type-safe slicing
arr.slice('0:2', ':');  // Returns NDArray
arr.get([0, 1]);        // Returns number
```

### Slicing Syntax

Since TypeScript doesn't support Python's `arr[0:5, :]` syntax, we use strings:

```typescript
// String-based (primary)
arr.slice('0:5', '1:3');     // arr[0:5, 1:3]
arr.slice(':', '-1');        // arr[:, -1]
arr.slice('::2');            // arr[::2]

// Convenience helpers
arr.row(0);                  // arr[0, :]
arr.col(2);                  // arr[:, 2]
arr.rows(0, 5);              // arr[0:5, :]
arr.cols(1, 3);              // arr[:, 1:3]
```

### Broadcasting

Automatic NumPy-style broadcasting:

```typescript
const a = np.ones([3, 4]);
const b = np.arange(4);
const c = a.add(b);  // (3, 4) + (4,) → (3, 4)
```

---

## Installation

```bash
npm install numpy-ts
```

```typescript
import * as np from 'numpy-ts';
```

---

## Documentation

### User Documentation
- [API-REFERENCE.md](https://github.com/dupontcyborg/numpy-ts/blob/main/docs/API-REFERENCE.md) - Complete API checklist

### Developer Documentation
- [TESTING-GUIDE.md](https://github.com/dupontcyborg/numpy-ts/blob/main/docs/TESTING-GUIDE.md) - How to add tests (unit, validation, benchmarks)
- [benchmarks/README.md](https://github.com/dupontcyborg/numpy-ts/blob/main/benchmarks/README.md) - Performance benchmarking guide

---

## Testing

Two-tier testing strategy:

1. **Unit Tests** - Test our implementation
2. **Python Comparison** - Validate against NumPy

```typescript
// Unit test
it('creates 2D array of zeros', () => {
  const arr = np.zeros([2, 3]);
  expect(arr.shape).toEqual([2, 3]);
  expect(arr.sum()).toBe(0);
});

// NumPy validation (cross-checked against Python)
it('matmul matches NumPy', () => {
  const A = np.array([[1, 2], [3, 4]]);
  const B = np.array([[5, 6], [7, 8]]);
  const result = A.matmul(B);

  const npResult = runNumPy(`
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = A @ B
  `);

  expect(result.toArray()).toEqual(npResult);
});

// Edge case validation
it('int8 overflow wraps like NumPy', () => {
  const arr = np.array([127], 'int8');
  const result = arr.add(1);
  expect(result.get([0])).toBe(-128);  // Wraps like NumPy
});
```

---

## Design Decisions

**Pure TypeScript Implementation**
- Built from scratch without heavy dependencies
- Focus on correctness and NumPy compatibility

**BigInt for int64/uint64**
- Exact representation over convenience
- No precision loss (different type but fully accurate)

**String-Based Slicing**
- `arr.slice('0:5', ':')` instead of `arr[0:5, :]`
- TypeScript limitation, Pythonic compromise

**View Tracking**
- Track base array with `base` attribute
- Zero-copy optimizations where possible
- Matches NumPy semantics

**Correctness First**
- Validate against Python NumPy before optimizing
- Performance improvements (WASM/SIMD) come later

---

## Contributing

Project is in early development. We welcome contributions!

### Setup

```bash
git clone https://github.com/dupontcyborg/numpy-ts.git
cd numpy-ts
npm install
npm test
```

### Adding New Features

1. Pick a function from [API-REFERENCE.md](https://github.com/dupontcyborg/numpy-ts/blob/main/docs/API-REFERENCE.md)
2. Follow the [TESTING-GUIDE.md](https://github.com/dupontcyborg/numpy-ts/blob/main/docs/TESTING-GUIDE.md) to add:
   - Implementation in `src/`
   - Unit tests in `tests/unit/`
   - NumPy validation tests in `tests/validation/`
   - Performance benchmarks in `benchmarks/`
3. Ensure all tests pass: `npm test`
4. Run benchmarks: `npm run bench:quick`
5. Submit a pull request

See [TESTING-GUIDE.md](https://github.com/dupontcyborg/numpy-ts/blob/main/docs/TESTING-GUIDE.md) for detailed instructions on adding tests.

---

## Comparison with Alternatives

| Feature | numpy-ts | numjs | ndarray | TensorFlow.js |
|---------|----------|-------|---------|---------------|
| API Coverage | 100% NumPy | ~20% | Different | ML-focused |
| TypeScript | Native | Partial | No | Yes |
| .npy files | Yes | No | No | No |
| Python-compatible | Yes | Mostly | No | No |
| Size | TBD | Small | Tiny | Large |

---

## Benchmarking

Compare numpy-ts performance against Python NumPy with auto-calibrated benchmarks:

```bash
# Quick benchmarks (~2-3 min) - 1 sample, 50ms/sample
source ~/.zshrc && conda activate py313 && npm run bench:quick

# Standard benchmarks (~5-10 min) - 5 samples, 100ms/sample (default)
source ~/.zshrc && conda activate py313 && npm run bench

# View interactive HTML report with ops/sec comparison
npm run bench:view
```

**Both modes use the same array sizes** - only sampling strategy differs (quick for speed, standard for accuracy).

### Performance Overview

![Benchmark Results](https://github.com/dupontcyborg/numpy-ts/blob/main/benchmarks/results/plots/latest.png)

See [benchmarks/README.md](https://github.com/dupontcyborg/numpy-ts/blob/main/benchmarks/README.md) for detailed benchmarking guide.

---

## Performance Expectations

**Current (v1.0)** - Pure TypeScript:
- 10-100x slower than NumPy
- Focus on correctness and API completeness

**Future (v2.0+)** - Optimizations:
- WASM for compute-intensive operations
- SIMD for vectorized operations
- Target: 2-20x slower than NumPy

Correctness and completeness first, then performance.

---

## License

[MIT License](https://github.com/dupontcyborg/numpy-ts/blob/main/LICENSE) - Copyright (c) 2025 Nicolas Dupont

---

## Links

- **NumPy**: https://numpy.org/
- **Issues**: https://github.com/dupontcyborg/numpy-ts/issues

---

**Ready to bring NumPy to TypeScript + JavaScript!** ⭐
