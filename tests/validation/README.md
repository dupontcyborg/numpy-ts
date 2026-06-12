# NumPy Validation Tests

These tests validate our TypeScript NumPy implementation against actual Python NumPy by executing Python code and comparing results.

## Requirements

- Python 3.x with NumPy installed

## Setup

### Option 1: System Python (Recommended)

```bash
pip install numpy
pnpm test
```

### Option 2: Conda Environment

```bash
conda install numpy
pnpm test
```

### Option 3: Custom Python Command

You can specify a custom Python command using the `NUMPY_PYTHON` environment variable:

```bash
# Using a specific conda environment
NUMPY_PYTHON="conda run -n myenv python" pnpm test

# Using a specific Python version
NUMPY_PYTHON="python3.11" pnpm test

# Using pyenv
NUMPY_PYTHON="pyenv exec python" pnpm test

# Using a virtual environment
NUMPY_PYTHON="/path/to/venv/bin/python" pnpm test
```

## Test Structure

### Validation Tests (require Python)

- **`exports.numpy.test.ts`** - Comprehensive API comparison
  - Tests each NumPy function individually (85+ functions)
  - Tests each ndarray method individually (40+ properties/methods)
  - Validates TS exports match Python NumPy's API
  - 149 tests: 20 passing (implemented), 129 failing (todo)

- **`creation.numpy.test.ts`** - Array creation functions
  - Tests: zeros, ones, array, arange, linspace, eye
  - 21 tests validating output matches Python exactly

- **`arithmetic.numpy.test.ts`** - Arithmetic operations
  - Tests: add, subtract, multiply, divide
  - 11 tests for scalar and array operations

- **`reductions.numpy.test.ts`** - Reduction operations
  - Tests: sum, mean, max, min
  - 15 tests with edge cases

- **`matmul.numpy.test.ts`** - Matrix multiplication
  - Tests matmul with various matrix sizes
  - 7 tests from 2x2 to 5x5 matrices

### Unit Tests (no Python required)

See `tests/unit/` for fast TypeScript-only tests.

## Running Tests

```bash
# All tests (requires Python with NumPy)
pnpm test

# Only validation tests
pnpm test tests/validation/

# Only unit tests (no Python required)
pnpm test tests/unit/

# Specific validation test
pnpm test tests/validation/arithmetic.numpy.test.ts

# With custom Python
NUMPY_PYTHON="python3.12" pnpm test
```

## Test Oracle

The `numpy-oracle.ts` module provides utilities for running Python code and comparing results:

- `runNumPy(code: string)` - Execute Python code and return result
- `arraysClose(a, b, rtol?, atol?)` - Compare arrays with floating-point tolerance
- `closeEnough(a, b, rtol?, atol?)` - Compare scalars with tolerance
- `checkNumPyAvailable()` - Check if Python + NumPy is available
- `getPythonInfo()` - Get Python and NumPy version info

### Tolerance

Floating-point comparisons use:
- `rtol = 1e-5` (relative tolerance)
- `atol = 1e-8` (absolute tolerance)

This matches NumPy's `allclose` defaults.

## Example Test

```typescript
import { array } from '../../src/core/ndarray';
import { runNumPy, arraysClose } from './numpy-oracle';

it('matches NumPy for scalar addition', () => {
  const jsResult = array([1, 2, 3]).add(10);
  const pyResult = runNumPy(`
result = np.array([1, 2, 3]) + 10
  `);

  expect(jsResult.shape).toEqual(pyResult.shape);
  expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
});
```

## Troubleshooting

### "Python NumPy not available"

1. Check Python is installed: `python3 --version`
2. Check NumPy is installed: `python3 -c "import numpy; print(numpy.__version__)"`
3. If using conda, ensure environment is activated or use `NUMPY_PYTHON`

### Wrong Python version being used

Set `NUMPY_PYTHON` to specify the exact Python command:

```bash
NUMPY_PYTHON="python3.11" pnpm test
```

### Tests pass in Python validation but fail in unit tests

This means the implementation is correct but might have TypeScript type issues or missing exports.
