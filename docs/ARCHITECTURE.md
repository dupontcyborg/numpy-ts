# Architecture

numpy-ts has a two-tier architecture designed for optimal tree-shaking while maintaining a convenient NumPy-like API.

## Directory Structure

```
src/
├── common/                    # Shared internals (not directly exported)
│   ├── ndarray-core.ts        # NDArrayCore class
│   ├── storage.ts             # ArrayStorage (typed arrays + shape metadata)
│   ├── dtype.ts               # DType utilities and promotion rules
│   ├── complex.ts             # Complex number implementation
│   ├── broadcasting.ts        # Broadcasting utilities
│   ├── slicing.ts             # Slice parsing and computation
│   └── ops/                   # Low-level ArrayStorage operations
│       ├── arithmetic.ts      # add, subtract, multiply, etc.
│       ├── shape.ts           # reshape, transpose, etc.
│       ├── reduction.ts       # sum, mean, max, etc.
│       ├── linalg.ts          # dot, matmul, etc.
│       └── ...
│
├── core/                      # Tree-shakeable entry (returns NDArrayCore)
│   ├── creation.ts            # zeros, ones, array, etc.
│   ├── arithmetic.ts          # add, subtract, multiply, etc.
│   ├── shape.ts               # reshape, transpose, etc.
│   ├── reduction.ts           # sum, mean, max, etc.
│   ├── linalg.ts              # dot, matmul, etc.
│   └── index.ts               # Re-exports everything
│
├── full/                      # Full library (returns NDArray)
│   ├── ndarray.ts             # NDArray class (extends NDArrayCore with methods)
│   └── index.ts               # Wrapper functions that upgrade NDArrayCore to NDArray
│
├── index.ts                   # Main entry: export * from './full'
├── core.ts                    # Core entry: export * from './core'
└── node.ts                    # Node.js file I/O functions
```

## Entry Points

| Import Path | Returns | Bundle Size | Use Case |
|-------------|---------|-------------|----------|
| `numpy-ts` | `NDArray` | ~180KB | Method chaining, NumPy-like syntax |
| `numpy-ts/core` | `NDArrayCore` | ~11-40KB | Tree-shaking, library code |
| `numpy-ts/node` | `NDArray` | ~180KB + fs | Node.js file I/O |

## NDArray vs NDArrayCore

### NDArrayCore (common/ndarray-core.ts)

Minimal array class with data storage only:

```typescript
class NDArrayCore {
  readonly _storage: ArrayStorage;

  constructor(storage: ArrayStorage) {
    this._storage = storage;
  }

  // Properties only
  get shape(): readonly number[] { ... }
  get dtype(): DType { ... }
  get data(): TypedArray { ... }
  get size(): number { ... }
  get ndim(): number { ... }
  get base(): NDArrayCore | null { ... }
  get flags(): Flags { ... }

  // Essential methods
  copy(): NDArrayCore { ... }
  astype(dtype: DType): NDArrayCore { ... }
  fill(value: number): void { ... }
  toArray(): NestedArray { ... }
}
```

### NDArray (full/ndarray.ts)

Full array class extending NDArrayCore with operation methods:

```typescript
class NDArray extends NDArrayCore {
  // Zero-copy upgrade from NDArrayCore
  static from(core: NDArrayCore): NDArray {
    return new NDArray(core._storage);
  }

  // All operation methods (~150 methods)
  add(other: NDArray | number): NDArray { ... }
  subtract(other: NDArray | number): NDArray { ... }
  reshape(shape: number[]): NDArray { ... }
  transpose(): NDArray { ... }
  sum(axis?: number): number | NDArray { ... }
  mean(axis?: number): number | NDArray { ... }
  sin(): NDArray { ... }
  cos(): NDArray { ... }
  dot(other: NDArray): NDArray { ... }
  matmul(other: NDArray): NDArray { ... }
  // ... many more
}
```

## Tree-Shaking

### How It Works

The `core/` functions return `NDArrayCore`, which has no operation methods attached. This allows bundlers to eliminate unused operations:

```typescript
// Minimal bundle (~11KB) - only zeros and array creation
import { zeros } from 'numpy-ts/core';
const a = zeros([3, 3]);
```

The `full/` functions wrap `core/` functions and upgrade results to `NDArray`:

```typescript
// Full bundle (~180KB) - includes all methods
import { zeros } from 'numpy-ts';
const a = zeros([3, 3]);
a.add(1).multiply(2).sum();  // Method chaining available
```

### Internal Upgrade Pattern

```typescript
// full/index.ts
import * as core from '../core';
import { NDArray } from './ndarray';

// Helper to upgrade NDArrayCore to NDArray
const up = (x: NDArrayCore): NDArray => NDArray.from(x);

export function zeros(shape: number[], dtype?: DType): NDArray {
  return up(core.zeros(shape, dtype));
}

export function add(a: ArrayLike, b: ArrayLike): NDArray {
  return up(core.add(a, b));
}
```

## Data Flow

```
User Code
    │
    ▼
full/index.ts (or core/index.ts)
    │
    ▼
core/*.ts functions
    │
    ▼
common/ops/*.ts (low-level operations on ArrayStorage)
    │
    ▼
common/storage.ts (ArrayStorage - typed arrays + shape metadata)
```

## Key Design Decisions

### 1. Single Source of Truth

All operations are implemented once in `common/ops/` or `core/`. The `full/` entry point wraps these with minimal overhead.

### 2. Zero-Copy Upgrades

`NDArray.from(core)` shares the same `ArrayStorage` - no data copying:

```typescript
const core = coreZeros([1000, 1000]);
const full = NDArray.from(core);
// core._storage === full._storage (same reference)
```

### 3. View Preservation

Functions that create views (transpose, reshape, slice) maintain the base reference:

```typescript
const a = zeros([4, 4]);
const b = a.transpose();
console.log(b.base === a);  // true - b is a view of a
```

### 4. Instanceof Compatibility

Both entry points create proper class instances:

```typescript
import { zeros } from 'numpy-ts';
import { zeros as coreZeros } from 'numpy-ts/core';

zeros([3, 3]) instanceof NDArray;      // true
coreZeros([3, 3]) instanceof NDArrayCore; // true
zeros([3, 3]) instanceof NDArrayCore;  // true (NDArray extends NDArrayCore)
```

## ArrayStorage

The internal data structure holding typed array data:

```typescript
class ArrayStorage {
  readonly data: TypedArray;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly strides: readonly number[];
  readonly offset: number;

  // Factory methods
  static zeros(shape: number[], dtype: DType): ArrayStorage;
  static ones(shape: number[], dtype: DType): ArrayStorage;
  static fromData(data: TypedArray, shape: number[], dtype: DType): ArrayStorage;

  // Operations
  copy(): ArrayStorage;
  reshape(newShape: number[]): ArrayStorage;
  broadcast(targetShape: number[]): ArrayStorage;
}
```

## DType System

NumPy-compatible data types with automatic promotion:

```typescript
type DType =
  | 'float64' | 'float32'
  | 'int64' | 'int32' | 'int16' | 'int8'
  | 'uint64' | 'uint32' | 'uint16' | 'uint8'
  | 'bool'
  | 'complex64' | 'complex128';

// Automatic promotion
promoteDTypes('int32', 'float32');  // 'float64'
promoteDTypes('int8', 'int16');     // 'int16'
```

## Broadcasting

NumPy-compatible broadcasting for element-wise operations:

```typescript
// (3, 4) + (4,) -> (3, 4)
const a = ones([3, 4]);
const b = arange(4);
const c = add(a, b);  // Each row gets b added

// (3, 1) * (1, 4) -> (3, 4)
const x = array([[1], [2], [3]]);  // (3, 1)
const y = array([[1, 2, 3, 4]]);   // (1, 4)
const z = multiply(x, y);          // (3, 4) outer product
```

## Testing Architecture

```
tests/
├── unit/                 # Fast unit tests (no Python)
│   ├── arithmetic.test.ts
│   ├── creation.test.ts
│   └── ...
│
├── validation/           # NumPy comparison tests (requires Python)
│   ├── arithmetic.numpy.test.ts
│   ├── creation.numpy.test.ts
│   └── numpy-oracle.ts   # Calls Python NumPy for expected values
│
├── bundles/              # Tests for bundled outputs
│   ├── esm.test.mjs
│   ├── node.test.cjs
│   └── browser.test.ts
│
└── tree-shaking/         # Bundle size verification
    └── tree-shaking.test.ts
```

## Build Output

```
dist/
├── numpy-ts.node.cjs      # Node.js CommonJS bundle
├── numpy-ts.browser.js    # Browser bundle
├── esm/                   # ES modules (tree-shakeable)
│   ├── index.js           # Full library entry
│   ├── core.js            # Core entry
│   └── ...
└── types/                 # TypeScript declarations
    ├── index.d.ts
    ├── core.d.ts
    └── ...
```
