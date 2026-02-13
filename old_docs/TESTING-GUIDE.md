# Testing Guide

Comprehensive guide for adding tests when implementing new functionality in numpy-ts.

## Table of Contents

- [Overview](#overview)
- [When to Add Tests](#when-to-add-tests)
- [Unit Tests](#unit-tests)
- [NumPy Validation Tests](#numpy-validation-tests)
- [Benchmarks](#benchmarks)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Best Practices](#best-practices)

---

## Overview

numpy-ts uses a **three-tier testing strategy**:

```
┌─────────────────────────────────────────────────────────┐
│                    Unit Tests                            │
│  Fast, isolated tests of individual functions           │
│  Location: tests/unit/                                   │
│  Purpose: Verify correctness, edge cases, error handling│
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│              NumPy Validation Tests                      │
│  Compare against Python NumPy for correctness           │
│  Location: tests/validation/                             │
│  Purpose: Ensure NumPy compatibility                     │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                    Benchmarks                            │
│  Performance comparison against NumPy                    │
│  Location: benchmarks/                                   │
│  Purpose: Track performance, identify bottlenecks        │
└─────────────────────────────────────────────────────────┘
```

**Golden Rule**: Every new feature or function must have all three types of tests.

---

## When to Add Tests

### Always Add Tests When:

✅ **Adding a new function** (e.g., `sqrt()`, `dot()`, `svd()`)
- Unit tests for basic functionality
- Validation tests against NumPy
- Benchmark comparing performance

✅ **Adding a new method** (e.g., `NDArray.sqrt()`)
- Same as above

✅ **Adding dtype support** (e.g., supporting `int32` in a function)
- Unit tests with new dtype
- Validation tests for dtype preservation
- Benchmark for dtype performance

✅ **Fixing a bug**
- Regression test to prevent bug from reoccurring
- Validation test if behavior differs from NumPy

✅ **Adding a new feature** (e.g., broadcasting, axis support)
- Comprehensive unit tests
- Validation tests for all cases
- Performance benchmarks

### Optional Tests:

⚠️ **Internal refactoring** - If behavior doesn't change, existing tests should pass
⚠️ **Documentation only** - No tests needed

---

## Unit Tests

### Location
`tests/unit/<feature>.test.ts`

Examples:
- `tests/unit/arithmetic.test.ts` - Arithmetic operations
- `tests/unit/reductions.test.ts` - Reduction operations
- `tests/unit/reshape.test.ts` - Shape manipulation

### Template

```typescript
/**
 * Tests for <feature>
 * Description of what this test file covers
 */

import { describe, it, expect } from 'vitest';
import { array, zeros, ones } from '../../src/core/ndarray';

describe('<Feature Name>', () => {
  describe('<sub-feature>', () => {
    it('handles basic case', () => {
      const arr = array([1, 2, 3]);
      const result = arr.myOperation();

      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([expected, values, here]);
    });

    it('handles edge case: empty array', () => {
      const arr = array([]);
      const result = arr.myOperation();

      expect(result.size).toBe(0);
    });

    it('handles edge case: scalar', () => {
      const arr = array([5]);
      const result = arr.myOperation();

      expect(result.get([0])).toBe(expectedValue);
    });

    it('throws on invalid input', () => {
      const arr = array([1, 2, 3]);

      expect(() => arr.myOperation(-1)).toThrow('Invalid argument');
    });

    it('preserves dtype', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = arr.myOperation();

      expect(result.dtype).toBe('int32');
    });

    it('works with 2D arrays', () => {
      const arr = array([[1, 2], [3, 4]]);
      const result = arr.myOperation();

      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([[...], [...]]);
    });
  });
});
```

### What to Test

1. **Basic functionality**
   ```typescript
   it('performs basic operation', () => {
     const arr = array([1, 2, 3]);
     const result = arr.sqrt();
     expect(result.toArray()).toBeCloseTo([1, 1.414, 1.732], 2);
   });
   ```

2. **Different array shapes**
   ```typescript
   it('works with 1D arrays', () => { /* ... */ });
   it('works with 2D arrays', () => { /* ... */ });
   it('works with 3D arrays', () => { /* ... */ });
   it('works with high-dimensional arrays', () => { /* ... */ });
   ```

3. **Different dtypes**
   ```typescript
   it('works with float64', () => { /* ... */ });
   it('works with int32', () => { /* ... */ });
   it('works with bool', () => { /* ... */ });
   ```

4. **Edge cases**
   ```typescript
   it('handles empty arrays', () => { /* ... */ });
   it('handles single element', () => { /* ... */ });
   it('handles large arrays', () => { /* ... */ });
   it('handles negative numbers', () => { /* ... */ });
   it('handles zero', () => { /* ... */ });
   ```

5. **Error handling**
   ```typescript
   it('throws on negative input for sqrt', () => {
     expect(() => array([-1]).sqrt()).toThrow();
   });
   ```

6. **Properties**
   ```typescript
   it('preserves dtype', () => { /* ... */ });
   it('returns correct shape', () => { /* ... */ });
   it('creates view when possible', () => {
     const arr = ones([3, 3]);
     const view = arr.transpose();
     expect(view.base).toBe(arr);
   });
   ```

### Example: Adding `sqrt()` Function

```typescript
// tests/unit/math.test.ts
import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src/core/ndarray';

describe('Math Operations', () => {
  describe('sqrt', () => {
    it('computes square root element-wise', () => {
      const arr = array([1, 4, 9, 16]);
      const result = arr.sqrt();

      expect(result.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('works with 2D arrays', () => {
      const arr = array([[1, 4], [9, 16]]);
      const result = arr.sqrt();

      expect(result.toArray()).toEqual([[1, 2], [3, 4]]);
    });

    it('handles floating point results', () => {
      const arr = array([2, 3, 5]);
      const result = arr.sqrt();

      expect(result.get([0])).toBeCloseTo(1.414, 3);
      expect(result.get([1])).toBeCloseTo(1.732, 3);
      expect(result.get([2])).toBeCloseTo(2.236, 3);
    });

    it('handles zero', () => {
      const arr = array([0]);
      const result = arr.sqrt();

      expect(result.get([0])).toBe(0);
    });

    it('preserves dtype for float types', () => {
      const arr = array([4], 'float32');
      const result = arr.sqrt();

      expect(result.dtype).toBe('float32');
    });

    it('converts int to float64', () => {
      const arr = array([4], 'int32');
      const result = arr.sqrt();

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(2);
    });

    it('throws on negative numbers', () => {
      const arr = array([-1]);

      expect(() => arr.sqrt()).toThrow('cannot compute sqrt of negative numbers');
    });
  });
});
```

---

## NumPy Validation Tests

### Location
`tests/validation/<feature>.numpy.test.ts`

Examples:
- `tests/validation/arithmetic.numpy.test.ts`
- `tests/validation/reductions.numpy.test.ts`
- `tests/validation/dtype-promotion.numpy.test.ts`

### Template

```typescript
/**
 * NumPy validation tests for <feature>
 * Validates against actual NumPy behavior
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, ones } from '../../src/core/ndarray';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: <Feature>', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  it('matches NumPy for basic operation', () => {
    const arr = array([1, 2, 3]);
    const result = arr.myOperation();

    const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.sqrt(arr)  # or whatever operation
    `);

    expect(result.toArray()).toEqual(npResult.value);
  });

  it('matches NumPy dtype behavior', () => {
    const arr = array([1, 2, 3], 'int32');
    const result = arr.myOperation();

    const npResult = runNumPy(`
arr = np.array([1, 2, 3], dtype=np.int32)
result = np.sqrt(arr)
    `);

    expect(result.dtype).toBe(npResult.dtype);
    expect(result.toArray()).toEqual(npResult.value);
  });
});
```

### What to Validate

1. **Basic behavior**
   ```typescript
   it('matches NumPy for 1D array', () => {
     const arr = array([1, 2, 3]);
     const result = arr.sqrt();

     const npResult = runNumPy(`
       arr = np.array([1, 2, 3])
       result = np.sqrt(arr)
     `);

     expect(result.toArray()).toEqual(npResult.value);
   });
   ```

2. **DType preservation and promotion**
   ```typescript
   it('matches NumPy dtype promotion', () => {
     const a = array([1], 'int8');
     const b = array([1], 'float32');
     const result = a.add(b);

     const npResult = runNumPy(`
       a = np.array([1], dtype=np.int8)
       b = np.array([1], dtype=np.float32)
       result = a + b
     `);

     expect(result.dtype).toBe(npResult.dtype);
   });
   ```

3. **Edge cases**
   ```typescript
   it('matches NumPy for NaN handling', () => {
     const arr = array([NaN, 1, 2]);
     const result = arr.sqrt();

     const npResult = runNumPy(`
       arr = np.array([np.nan, 1, 2])
       result = np.sqrt(arr)
     `);

     expect(Number.isNaN(result.get([0]))).toBe(true);
     expect(result.get([1])).toBe(npResult.value[1]);
   });
   ```

4. **Error behavior**
   ```typescript
   it('throws same error as NumPy', () => {
     // Test that we throw errors for the same conditions as NumPy
     const arr = array([1, 2]);

     expect(() => arr.reshape(3, 4)).toThrow();
     // NumPy also throws: ValueError: cannot reshape array of size 2 into shape (3,4)
   });
   ```

### Example: Adding `sqrt()` Validation

```typescript
// tests/validation/math.numpy.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src/core/ndarray';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: Math Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('sqrt', () => {
    it('matches NumPy for integers', () => {
      const arr = array([1, 4, 9, 16]);
      const result = arr.sqrt();

      const npResult = runNumPy(`
arr = np.array([1, 4, 9, 16])
result = np.sqrt(arr)
      `);

      expect(result.toArray()).toEqual(npResult.value);
      expect(result.dtype).toBe(npResult.dtype);
    });

    it('matches NumPy for floats', () => {
      const arr = array([2.0, 3.0, 5.0]);
      const result = arr.sqrt();

      const npResult = runNumPy(`
arr = np.array([2.0, 3.0, 5.0])
result = np.sqrt(arr)
      `);

      for (let i = 0; i < 3; i++) {
        expect(result.get([i])).toBeCloseTo(npResult.value[i], 10);
      }
    });

    it('matches NumPy dtype promotion for int32', () => {
      const arr = array([4], 'int32');
      const result = arr.sqrt();

      const npResult = runNumPy(`
arr = np.array([4], dtype=np.int32)
result = np.sqrt(arr)
      `);

      expect(result.dtype).toBe(npResult.dtype);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('matches NumPy for 2D arrays', () => {
      const arr = array([[1, 4], [9, 16]]);
      const result = arr.sqrt();

      const npResult = runNumPy(`
arr = np.array([[1, 4], [9, 16]])
result = np.sqrt(arr)
      `);

      expect(result.toArray()).toEqual(npResult.value);
    });

    it('matches NumPy handling of NaN', () => {
      const arr = array([NaN, 4]);
      const result = arr.sqrt();

      // Can't serialize NaN through JSON, so just verify behavior
      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(result.get([1])).toBe(2);
    });
  });
});
```

---

## Benchmarks

### Location
`benchmarks/src/specs.ts` - Add new benchmark specifications

### Template

```typescript
// In benchmarks/src/specs.ts, add to getBenchmarkSpecs():

specs.push({
  name: `operation_name [${sizes.medium.join('x')}]`,
  category: 'category_name',  // 'math', 'linalg', 'creation', etc.
  operation: 'operation_name',
  setup: {
    a: { shape: sizes.medium, fill: 'arange', dtype: 'float64' },
    // Add more arrays as needed
  },
  iterations,
  warmup
});
```

### Add Operation Support

1. **TypeScript runner** (`benchmarks/src/runner.ts`):
```typescript
function executeOperation(operation: string, arrays: Record<string, any>): any {
  // ... existing operations ...

  // Add your operation
  else if (operation === 'sqrt') {
    return arrays['a'].sqrt();
  }

  throw new Error(`Unknown operation: ${operation}`);
}
```

2. **Python runner** (`benchmarks/scripts/numpy_benchmark.py`):
```python
def execute_operation(operation: str, arrays: Dict[str, np.ndarray]) -> Any:
    # ... existing operations ...

    # Add your operation
    elif operation == 'sqrt':
        return np.sqrt(arrays['a'])

    raise ValueError(f"Unknown operation: {operation}")
```

### Example: Adding `sqrt()` Benchmark

```typescript
// In benchmarks/src/specs.ts

// Add to appropriate section
if (Array.isArray(sizes.medium)) {
  specs.push({
    name: `sqrt [${sizes.medium.join('x')}]`,
    category: 'math',
    operation: 'sqrt',
    setup: {
      a: { shape: sizes.medium, fill: 'arange', dtype: 'float64' }
    },
    iterations,
    warmup
  });
}

// For different dtypes
specs.push({
  name: `sqrt [${sizes.medium.join('x')}] float32`,
  category: 'math',
  operation: 'sqrt',
  setup: {
    a: { shape: sizes.medium, fill: 'arange', dtype: 'float32' }
  },
  iterations,
  warmup
});
```

```typescript
// In benchmarks/src/runner.ts
function executeOperation(operation: string, arrays: Record<string, any>): any {
  // ... existing code ...

  // Math operations
  else if (operation === 'sqrt') {
    return arrays['a'].sqrt();
  }
  else if (operation === 'exp') {
    return arrays['a'].exp();
  }
  else if (operation === 'log') {
    return arrays['a'].log();
  }

  throw new Error(`Unknown operation: ${operation}`);
}
```

```python
# In benchmarks/scripts/numpy_benchmark.py
def execute_operation(operation: str, arrays: Dict[str, np.ndarray]) -> Any:
    # ... existing code ...

    # Math operations
    elif operation == 'sqrt':
        return np.sqrt(arrays['a'])
    elif operation == 'exp':
        return np.exp(arrays['a'])
    elif operation == 'log':
        return np.log(arrays['a'])

    raise ValueError(f"Unknown operation: {operation}")
```

---

## Test Organization

### File Structure

```
tests/
├── unit/                           # Fast, isolated tests
│   ├── arithmetic.test.ts          # +, -, *, /
│   ├── comparisons.test.ts         # >, <, ==, !=
│   ├── creation.test.ts            # zeros, ones, arange
│   ├── dtype.test.ts               # DType system
│   ├── indexing.test.ts            # Slicing, indexing
│   ├── math.test.ts                # sqrt, exp, log, sin, cos
│   ├── ndarray.test.ts             # Core NDArray functionality
│   ├── reductions.test.ts          # sum, mean, max, min
│   ├── reshape.test.ts             # reshape, flatten, transpose
│   ├── view-tracking.test.ts       # base, flags
│   └── ...                         # Add more as needed
│
└── validation/                     # NumPy comparison tests
    ├── arithmetic.numpy.test.ts
    ├── comparisons.numpy.test.ts
    ├── dtype-edge-cases.numpy.test.ts
    ├── dtype-promotion.numpy.test.ts
    ├── math.numpy.test.ts          # Add when implementing math ops
    ├── matmul.numpy.test.ts
    ├── reductions.numpy.test.ts
    └── ...                         # Add more as needed
```

### Naming Conventions

- **Unit tests**: `<feature>.test.ts`
- **Validation tests**: `<feature>.numpy.test.ts`
- **Test suites**: Use descriptive `describe()` blocks
- **Test cases**: Start with lowercase, be specific

**Good**:
```typescript
describe('Math Operations', () => {
  describe('sqrt', () => {
    it('computes square root element-wise', () => { /* ... */ });
    it('throws on negative input', () => { /* ... */ });
  });
});
```

**Bad**:
```typescript
describe('Test sqrt', () => {
  it('test 1', () => { /* ... */ });
  it('test 2', () => { /* ... */ });
});
```

---

## Running Tests

### All Tests
```bash
npm test                    # Run all tests
npm run test:quick          # Skip slow validation tests
```

### Unit Tests Only
```bash
npm run test:unit           # All unit tests
npm run test:unit -- math   # Just math.test.ts
```

### Validation Tests Only
```bash
npm run test:validation           # All validation tests
npm run test:validation -- math   # Just math.numpy.test.ts
```

### Watch Mode
```bash
npm run test:watch          # Re-run on file changes
```

### Coverage
```bash
npm run test:coverage       # Generate coverage report
```

### Benchmarks
```bash
npm run bench:quick         # Quick benchmarks
npm run bench               # Standard benchmarks
npm run bench:category math # Just math benchmarks
```

---

## Best Practices

### 1. **Test Independence**
Each test should be independent and not rely on other tests.

**Good**:
```typescript
it('test A', () => {
  const arr = array([1, 2, 3]);
  expect(arr.sum()).toBe(6);
});

it('test B', () => {
  const arr = array([1, 2, 3]);
  expect(arr.mean()).toBe(2);
});
```

**Bad**:
```typescript
let sharedArr;

it('test A', () => {
  sharedArr = array([1, 2, 3]);
  expect(sharedArr.sum()).toBe(6);
});

it('test B', () => {
  // Relies on test A running first
  expect(sharedArr.mean()).toBe(2);
});
```

### 2. **Clear Test Names**
Test names should clearly describe what is being tested.

**Good**:
```typescript
it('returns correct shape for 2D array', () => { /* ... */ });
it('throws error for negative axis value', () => { /* ... */ });
it('preserves int32 dtype after addition', () => { /* ... */ });
```

**Bad**:
```typescript
it('works', () => { /* ... */ });
it('test 1', () => { /* ... */ });
it('check shape', () => { /* ... */ });
```

### 3. **Use Appropriate Matchers**

```typescript
// For exact equality
expect(result).toBe(5);
expect(result).toEqual([1, 2, 3]);

// For floating point
expect(result).toBeCloseTo(3.14159, 5);  // 5 decimal places

// For arrays
expect(arr.toArray()).toEqual([[1, 2], [3, 4]]);

// For errors
expect(() => arr.invalid()).toThrow();
expect(() => arr.invalid()).toThrow('specific message');

// For NaN
expect(Number.isNaN(result)).toBe(true);
```

### 4. **Test Edge Cases**

Always test:
- Empty arrays: `array([])`
- Single element: `array([5])`
- Large arrays: `ones([1000, 1000])`
- Negative numbers
- Zero
- NaN, Infinity, -Infinity (for float operations)
- Minimum/maximum dtype values

### 5. **Document Non-Obvious Behavior**

```typescript
it('returns view for C-contiguous arrays', () => {
  // reshape() returns a view when the array is C-contiguous
  // and can be reshaped without copying data
  const arr = ones([2, 6]);  // C-contiguous
  const reshaped = arr.reshape(3, 4);

  expect(reshaped.base).toBe(arr);  // It's a view
  expect(reshaped.flags.OWNDATA).toBe(false);
});
```

### 6. **Group Related Tests**

```typescript
describe('Math Operations', () => {
  describe('sqrt', () => {
    it('basic functionality', () => { /* ... */ });
    it('edge cases', () => { /* ... */ });
  });

  describe('exp', () => {
    it('basic functionality', () => { /* ... */ });
    it('edge cases', () => { /* ... */ });
  });
});
```

### 7. **Keep Tests Fast**

- Unit tests should be < 10ms each
- Use small arrays for unit tests (< 100 elements)
- Use larger arrays only for benchmarks
- Skip slow tests in quick mode

### 8. **Validation Tests Should Match NumPy Exactly**

```typescript
it('matches NumPy exactly', () => {
  const arr = array([[1, 2], [3, 4]]);
  const result = arr.sum(0);

  const npResult = runNumPy(`
arr = np.array([[1, 2], [3, 4]])
result = arr.sum(axis=0)
  `);

  // Check both value and dtype
  expect(result.toArray()).toEqual(npResult.value);
  expect(result.dtype).toBe(npResult.dtype);
});
```

### 9. **Benchmarks Should Be Representative**

- Test realistic array sizes (not just tiny arrays)
- Include multiple sizes (small, medium, large)
- Test both fast and slow paths
- Include different dtypes if relevant

### 10. **Update Documentation**

When adding tests, also update:
- `API-REFERENCE.md` - Mark function as complete
- `README.md` - Update feature list if needed
- This guide - If you discover new patterns

---

## Complete Example: Adding a New Function

Let's walk through adding a complete `log()` function with all tests.

### Step 1: Unit Tests

```typescript
// tests/unit/math.test.ts
describe('Math Operations', () => {
  describe('log', () => {
    it('computes natural logarithm element-wise', () => {
      const arr = array([1, Math.E, Math.E ** 2]);
      const result = arr.log();

      expect(result.get([0])).toBeCloseTo(0, 10);
      expect(result.get([1])).toBeCloseTo(1, 10);
      expect(result.get([2])).toBeCloseTo(2, 10);
    });

    it('works with 2D arrays', () => {
      const arr = array([[1, Math.E], [Math.E ** 2, Math.E ** 3]]);
      const result = arr.log();

      expect(result.get([0, 0])).toBeCloseTo(0, 10);
      expect(result.get([0, 1])).toBeCloseTo(1, 10);
      expect(result.get([1, 0])).toBeCloseTo(2, 10);
      expect(result.get([1, 1])).toBeCloseTo(3, 10);
    });

    it('handles log(1) = 0', () => {
      const arr = array([1]);
      const result = arr.log();

      expect(result.get([0])).toBe(0);
    });

    it('returns -Infinity for log(0)', () => {
      const arr = array([0]);
      const result = arr.log();

      expect(result.get([0])).toBe(-Infinity);
    });

    it('returns NaN for negative numbers', () => {
      const arr = array([-1]);
      const result = arr.log();

      expect(Number.isNaN(result.get([0]))).toBe(true);
    });

    it('preserves float dtype', () => {
      const arr = array([Math.E], 'float32');
      const result = arr.log();

      expect(result.dtype).toBe('float32');
    });

    it('converts int to float64', () => {
      const arr = array([1], 'int32');
      const result = arr.log();

      expect(result.dtype).toBe('float64');
    });
  });
});
```

### Step 2: Validation Tests

```typescript
// tests/validation/math.numpy.test.ts
describe('NumPy Validation: Math Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('log', () => {
    it('matches NumPy for positive numbers', () => {
      const arr = array([1, 2, Math.E, 10, 100]);
      const result = arr.log();

      const npResult = runNumPy(`
arr = np.array([1, 2, ${Math.E}, 10, 100])
result = np.log(arr)
      `);

      for (let i = 0; i < 5; i++) {
        expect(result.get([i])).toBeCloseTo(npResult.value[i], 10);
      }
    });

    it('matches NumPy for 2D arrays', () => {
      const arr = array([[1, 2], [Math.E, 10]]);
      const result = arr.log();

      const npResult = runNumPy(`
arr = np.array([[1, 2], [${Math.E}, 10]])
result = np.log(arr)
      `);

      expect(result.toArray()[0]![0]).toBeCloseTo(npResult.value[0][0], 10);
      expect(result.toArray()[0]![1]).toBeCloseTo(npResult.value[0][1], 10);
      expect(result.toArray()[1]![0]).toBeCloseTo(npResult.value[1][0], 10);
      expect(result.toArray()[1]![1]).toBeCloseTo(npResult.value[1][1], 10);
    });

    it('matches NumPy dtype conversion', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = arr.log();

      const npResult = runNumPy(`
arr = np.array([1, 2, 3], dtype=np.int32)
result = np.log(arr)
      `);

      expect(result.dtype).toBe(npResult.dtype);
    });

    it('matches NumPy for edge cases', () => {
      const arr = array([0, 1]);
      const result = arr.log();

      // log(0) = -inf, log(1) = 0 in both NumPy and JavaScript
      expect(result.get([0])).toBe(-Infinity);
      expect(result.get([1])).toBe(0);
    });
  });
});
```

### Step 3: Benchmarks

```typescript
// In benchmarks/src/specs.ts

// Add to math operations section
if (Array.isArray(sizes.medium)) {
  specs.push({
    name: `log [${sizes.medium.join('x')}]`,
    category: 'math',
    operation: 'log',
    setup: {
      a: { shape: sizes.medium, fill: 'arange', dtype: 'float64' }
    },
    iterations,
    warmup
  });
}

// In benchmarks/src/runner.ts - add to executeOperation():
else if (operation === 'log') {
  return arrays['a'].log();
}

// In benchmarks/scripts/numpy_benchmark.py - add to execute_operation():
elif operation == 'log':
    return np.log(arrays['a'])
```

### Step 4: Run Tests

```bash
# Run unit tests
npm run test:unit -- math

# Run validation tests
npm run test:validation -- math

# Run benchmarks
npm run bench:category math
```

### Step 5: Update Documentation

Update `docs/API-REFERENCE.md`:
```markdown
### Mathematical Functions
- [x] `log(x)` - Natural logarithm _(implemented as NDArray.log() method)_
```

---

## Summary Checklist

When adding a new feature, check off:

- [ ] Implementation in `src/`
- [ ] Unit tests in `tests/unit/`
  - [ ] Basic functionality
  - [ ] Edge cases
  - [ ] Error handling
  - [ ] DType preservation
  - [ ] Different array shapes
- [ ] Validation tests in `tests/validation/`
  - [ ] Compare against NumPy
  - [ ] Verify dtype behavior
  - [ ] Test edge cases
- [ ] Benchmarks
  - [ ] Add to `benchmarks/src/specs.ts`
  - [ ] Add to `benchmarks/src/runner.ts`
  - [ ] Add to `benchmarks/scripts/numpy_benchmark.py`
- [ ] Documentation
  - [ ] Update `API-REFERENCE.md`
  - [ ] Add JSDoc comments
  - [ ] Update README if major feature

---

**Remember**: Testing is not optional. Every function must have unit tests, validation tests, and benchmarks. This ensures correctness, NumPy compatibility, and helps track performance improvements over time.
