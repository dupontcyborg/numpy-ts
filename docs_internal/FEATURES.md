# Feature Details

In-depth documentation of numpy-ts features and capabilities.

## Table of Contents

- [Data Types (dtypes)](#data-types-dtypes)
- [Broadcasting](#broadcasting)
- [View Semantics](#view-semantics)
- [Memory Layout](#memory-layout)
- [Slicing](#slicing)
- [Type Promotion](#type-promotion)

---

## Data Types (dtypes)

numpy-ts supports 11 NumPy-compatible data types:

### Floating Point
- `'float64'` (default) - 64-bit IEEE 754 floating point
- `'float32'` - 32-bit IEEE 754 floating point

### Signed Integers
- `'int64'` - 64-bit signed integer (uses BigInt)
- `'int32'` - 32-bit signed integer
- `'int16'` - 16-bit signed integer
- `'int8'` - 8-bit signed integer

### Unsigned Integers
- `'uint64'` - 64-bit unsigned integer (uses BigInt)
- `'uint32'` - 32-bit unsigned integer
- `'uint16'` - 16-bit unsigned integer
- `'uint8'` - 8-bit unsigned integer

### Boolean
- `'bool'` - Boolean (stored as uint8)

### Creating Arrays with dtypes

```typescript
import * as np from 'numpy-ts';

// Default is float64
const a = np.zeros([3, 3]);  // float64

// Specify dtype
const b = np.ones([3, 3], 'int32');
const c = np.arange(0, 10, 1, 'float32');
const d = np.array([1, 0, 1], 'bool');

// BigInt support for int64/uint64
const e = np.array([1n, 2n, 3n], 'int64');
```

### DType Preservation

All operations preserve dtype or follow NumPy promotion rules:

```typescript
const intArr = np.ones([3, 3], 'int32');
const result = intArr.add(5);  // Stays int32

const floatArr = np.ones([3, 3], 'float32');
const promoted = intArr.add(floatArr);  // Promotes to float32
```

### Overflow Behavior

Integer dtypes wrap on overflow, matching NumPy:

```typescript
const arr = np.array([127], 'int8');
const result = arr.add(1);
console.log(result.get([0]));  // -128 (wrapped)
```

---

## Broadcasting

Broadcasting allows operations on arrays with different but compatible shapes. numpy-ts implements full NumPy broadcasting semantics.

### Broadcasting Rules

1. If arrays have different ranks, prepend shape of smaller array with 1s
2. Arrays are compatible when dimensions are equal or one of them is 1
3. Output shape is the maximum along each dimension

### Examples

#### Scalar Broadcasting
```typescript
const arr = np.array([[1, 2, 3], [4, 5, 6]]);  // (2, 3)
const result = arr.add(10);  // Scalar broadcasts to (2, 3)
// [[11, 12, 13], [14, 15, 16]]
```

#### Vector Broadcasting
```typescript
const matrix = np.ones([3, 4]);  // (3, 4)
const row = np.arange(4);        // (4,)
const result = matrix.add(row);   // row broadcasts to (3, 4)
// Each row gets the vector added
```

#### Matrix Broadcasting
```typescript
const row = np.array([1, 2, 3, 4]);        // (4,)
const col = np.array([[1], [2], [3]]);     // (3, 1)
const grid = col.multiply(row);             // (3, 4)
// Outer product via broadcasting
```

#### Advanced Broadcasting
```typescript
const a = np.ones([3, 1, 4]);  // (3, 1, 4)
const b = np.ones([2, 1]);     // (2, 1)
const c = a.add(b);            // (3, 2, 4)
```

### Explicit Broadcasting

```typescript
import * as np from 'numpy-ts';

// broadcast_to - broadcast to specific shape
const arr = np.array([1, 2, 3]);  // (3,)
const broadcasted = np.broadcast_to(arr, [4, 3]);  // (4, 3)

// broadcast_arrays - broadcast multiple arrays to common shape
const [a, b] = np.broadcast_arrays(
  np.ones([3, 1]),
  np.ones([1, 4])
);  // Both become (3, 4)
```

---

## View Semantics

numpy-ts implements NumPy's view semantics for memory-efficient operations.

### What is a View?

A view is an array that shares data with another array but has its own shape, strides, and offset. Modifying a view modifies the underlying data.

### Operations That Return Views

- `slice()` - Slicing (when possible)
- `transpose()` - Always returns a view
- `swapaxes()` - Always returns a view
- `moveaxis()` - Always returns a view
- `squeeze()` - Always returns a view
- `expand_dims()` - Always returns a view
- `reshape()` - View if C-contiguous, copy otherwise
- `ravel()` - View if C-contiguous, copy otherwise
- `broadcast_to()` - Always returns a view

### Operations That Return Copies

- `flatten()` - Always returns a copy
- `copy()` - Explicit copy
- Any operation that cannot be represented with strides

### Tracking Views

```typescript
const arr = np.ones([4, 4]);
const view = arr.slice('0:2', '0:2');

// Check if it's a view
console.log(view.base === arr);       // true - view tracks base
console.log(view.flags.OWNDATA);      // false - doesn't own data
console.log(arr.flags.OWNDATA);       // true - owns data
```

### Modifying Views

```typescript
const arr = np.array([[1, 2], [3, 4]]);
const view = arr.slice('0:1', ':');

// Modifying view modifies original
view.set([0, 0], 99);
console.log(arr.get([0, 0]));  // 99
```

---

## Memory Layout

numpy-ts tracks memory layout with flags, enabling optimizations.

### Layout Flags

- `C_CONTIGUOUS` - Row-major (C-style) contiguous memory
- `F_CONTIGUOUS` - Column-major (Fortran-style) contiguous memory
- `OWNDATA` - Array owns its data (not a view)

### Checking Layout

```typescript
const arr = np.ones([3, 4]);

console.log(arr.flags.C_CONTIGUOUS);  // true - row-major
console.log(arr.flags.F_CONTIGUOUS);  // false (only [1,1] can be both)
console.log(arr.flags.OWNDATA);       // true - owns data

const transposed = arr.transpose();
console.log(transposed.flags.C_CONTIGUOUS);  // false
console.log(transposed.flags.F_CONTIGUOUS);  // true - now column-major
console.log(transposed.flags.OWNDATA);       // false - doesn't own data
```

### Why Layout Matters

Operations like `reshape()` and `ravel()` can return views only if the array is C-contiguous:

```typescript
const arr = np.ones([2, 3]);
const reshaped = arr.reshape([3, 2]);  // View (C-contiguous)

const transposed = arr.transpose();
const reshapedT = transposed.reshape([3, 2]);  // Copy (not C-contiguous)
```

---

## Slicing

TypeScript doesn't support Python's `arr[0:5, :]` syntax, so numpy-ts uses strings.

### Basic Slicing

```typescript
// Single dimension
arr.slice(':')        // arr[:]
arr.slice('0:5')      // arr[0:5]
arr.slice('::2')      // arr[::2]
arr.slice('::-1')     // arr[::-1] (reverse)
arr.slice('-1')       // arr[-1] (last element)

// Multiple dimensions
arr.slice('0:5', ':')           // arr[0:5, :]
arr.slice(':', '1:3')           // arr[:, 1:3]
arr.slice('::2', '1:')          // arr[::2, 1:]
arr.slice('0:10:2', '5:10:1')   // arr[0:10:2, 5:10:1]
```

### Convenience Methods

```typescript
// Single row/column
arr.row(0)      // arr[0, :]
arr.col(2)      // arr[:, 2]

// Row/column ranges
arr.rows(0, 5)  // arr[0:5, :]
arr.cols(1, 3)  // arr[:, 1:3]
```

### Negative Indices

```typescript
arr.slice('-1', ':')     // Last row
arr.slice(':', '-1')     // Last column
arr.slice('-3:', ':')    // Last 3 rows
```

### Getting/Setting Elements

```typescript
// Get single element
const val = arr.get([0, 1]);

// Set single element
arr.set([0, 1], 99);
```

---

## Type Promotion

When operating on arrays with different dtypes, numpy-ts follows NumPy's type promotion rules.

### Promotion Hierarchy

```
float64 > float32 > int64 > int32 > int16 > int8 >
uint64 > uint32 > uint16 > uint8 > bool
```

### Examples

```typescript
// int32 + float32 → float32
const a = np.ones([3], 'int32');
const b = np.ones([3], 'float32');
const c = a.add(b);  // float32

// uint8 + int16 → int16
const d = np.ones([3], 'uint8');
const e = np.ones([3], 'int16');
const f = d.add(e);  // int16

// bool + int8 → int8
const g = np.ones([3], 'bool');
const h = np.ones([3], 'int8');
const i = g.add(h);  // int8
```

### Special Cases

- **Comparisons** always return `'bool'` dtype
- **Reductions** preserve dtype except `mean()` converts integers to `'float64'`

```typescript
const arr = np.ones([3, 3], 'int32');

// Comparison returns bool
const mask = arr.greater(0);  // bool dtype

// Sum preserves dtype
const total = arr.sum();  // int32

// Mean converts to float64
const avg = arr.mean();  // float64
```

---

## Additional Resources

- [API Reference](API-REFERENCE.md) - Complete function list
- [Architecture](ARCHITECTURE.md) - System design
- [Testing Guide](TESTING-GUIDE.md) - How to add tests
