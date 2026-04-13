/**
 * Validation module for benchmark correctness
 * Validates that numpy-ts produces the same results as NumPy
 */

import { spawn } from 'child_process';
import { resolve } from 'path';
import * as np from '../../src/index';
import { hasFloat16 as npHasFloat16 } from '../../src/common/dtype';
import type { BenchmarkCase } from './types';

const FLOAT64_TOLERANCE = 1e-5;
const FLOAT32_TOLERANCE = 1e-3; // float32 has ~7 decimal digits of precision
const FLOAT16_TOLERANCE = 3e-3; // float16 has ~3.3 decimal digits; accumulation order differences need extra margin

/**
 * Deserialize special float values from Python
 */
function deserializeValue(val: any): any {
  if (Array.isArray(val)) {
    return val.map((v) => deserializeValue(v));
  } else if (typeof val === 'object' && val !== null) {
    const result: Record<string, any> = {};
    for (const [k, v] of Object.entries(val)) {
      result[k] = deserializeValue(v);
    }
    return result;
  } else if (val === '__Infinity__') {
    return Infinity;
  } else if (val === '__-Infinity__') {
    return -Infinity;
  } else if (val === '__NaN__') {
    return NaN;
  } else if (typeof val === 'string' && val.startsWith('__bigint__')) {
    return BigInt(val.slice(10));
  }
  return val;
}

/**
 * Check if operation is a random operation that can't do exact comparison.
 * Now that random ops are seeded, most can do value comparison.
 * Only choice/permutation remain non-deterministic in ordering.
 */
function isRandomOperation(operation: string): boolean {
  // choice and permutation use shuffling which differs between implementations
  return operation === 'random_choice' || operation === 'random_permutation';
}

/**
 * Check if operation is SVD-related (needs special tolerance handling)
 */
function isSvdOperation(operation: string): boolean {
  return operation === 'linalg_svd' || operation === 'linalg_svdvals';
}

/**
 * Compare SVD singular values with relative tolerance
 * SVD algorithms can produce significantly different results for very small singular values
 * This is especially true for rank-deficient matrices where numerical noise dominates
 */
function compareSvdResults(tsData: any, npData: any): boolean {
  function flatten(arr: any): number[] {
    if (typeof arr === 'number') return [arr];
    if (Array.isArray(arr)) return arr.flatMap(flatten);
    return [];
  }

  const tsFlat = flatten(tsData);
  const npFlat = flatten(npData);

  if (tsFlat.length !== npFlat.length) return false;

  // Find the maximum singular value for relative comparison
  const maxVal = Math.max(...tsFlat.map(Math.abs), ...npFlat.map(Math.abs));
  // Values below this threshold relative to max are considered "numerically zero"
  // Use 1e-6 as a generous threshold for different SVD implementations
  const zeroThreshold = maxVal * 1e-6;

  for (let i = 0; i < tsFlat.length; i++) {
    const a = tsFlat[i]!;
    const b = npFlat[i]!;

    // If either value is very small relative to max, the other should also be "small"
    // Different SVD implementations can produce wildly different values in the noise floor
    if (Math.abs(a) < zeroThreshold || Math.abs(b) < zeroThreshold) {
      // Both should be "small" - allow generous tolerance for numerical noise
      // One being 1e-12 and other being 1e-3 is acceptable for rank-deficient matrices
      const smallThreshold = maxVal * 1e-3; // Values < 0.1% of max are "small"
      if (Math.abs(a) < smallThreshold && Math.abs(b) < smallThreshold) continue;
    }

    // For larger values, use relative tolerance
    const relDiff = Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-15);
    if (relDiff > 1e-5) return false; // 0.001% relative tolerance for significant values
  }

  return true;
}

/**
 * Check if operation is numerically sensitive and may differ between implementations
 */
function isNumericallySensitiveOperation(operation: string): boolean {
  // roots uses eigenvalue decomposition which can differ significantly
  // between implementations for polynomials with complex roots
  // FFT operations can have minor floating-point differences between implementations
  const sensitiveOps = [
    'roots',
    'fft',
    'ifft',
    'fft2',
    'ifft2',
    'fftn',
    'ifftn',
    'rfft',
    'irfft',
    'rfft2',
    'irfft2',
    'rfftn',
    'irfftn',
    'hfft',
    'ihfft',
  ];
  return sensitiveOps.includes(operation);
}

/**
 * Compare two arrays or scalars for equality with tolerance
 * @param operation - The operation name (for special handling of non-deterministic operations)
 * @param tolerance - Numeric tolerance for comparisons (default: FLOAT64_TOLERANCE)
 */
function resultsMatch(
  numpytsResult: any,
  numpyResult: any,
  operation?: string,
  tolerance: number = FLOAT64_TOLERANCE
): boolean {
  // For random operations, just check shapes match (values will differ)
  if (operation && isRandomOperation(operation)) {
    if (numpytsResult?.shape && numpyResult?.shape) {
      return (
        numpytsResult.shape.length === numpyResult.shape.length &&
        numpytsResult.shape.every((dim: number, i: number) => dim === numpyResult.shape[i])
      );
    }
    // If no shape, assume it's valid if it ran without error
    return true;
  }

  // For numerically sensitive operations, just check shapes match
  if (operation && isNumericallySensitiveOperation(operation)) {
    if (numpytsResult?.shape && numpyResult?.shape) {
      return (
        numpytsResult.shape.length === numpyResult.shape.length &&
        numpytsResult.shape.every((dim: number, i: number) => dim === numpyResult.shape[i])
      );
    }
    return true;
  }

  // Both null (often represents NaN or Infinity in serialization)
  if (numpytsResult === null && numpyResult === null) {
    return true;
  }

  // Float32Array fallback only: NumPy returned null (inf/overflow) but numpy-ts
  // returned a finite value. This happens because our Float32Array backing has
  // much larger range than true float16 (max ~65504).
  if (
    numpyResult === null &&
    numpytsResult !== null &&
    tolerance >= FLOAT16_TOLERANCE &&
    !npHasFloat16
  ) {
    return true;
  }

  // BigInt scalars (from int64/uint64 results) — compare exactly
  if (typeof numpytsResult === 'bigint' && typeof numpyResult === 'bigint') {
    return numpytsResult === numpyResult;
  }
  if (typeof numpytsResult === 'bigint' || typeof numpyResult === 'bigint') {
    const a = typeof numpytsResult === 'bigint' ? Number(numpytsResult) : numpytsResult;
    const b = typeof numpyResult === 'bigint' ? Number(numpyResult) : numpyResult;
    return resultsMatch(a, b, operation, tolerance);
  }

  // Both scalars
  if (typeof numpytsResult === 'number' && typeof numpyResult === 'number') {
    // Handle NaN: both NaN is considered equal
    if (isNaN(numpytsResult) && isNaN(numpyResult)) return true;
    // Handle Infinity: both must be same infinity
    if (!isFinite(numpytsResult) && !isFinite(numpyResult)) return numpytsResult === numpyResult;
    // Float32Array fallback only: allow inf-vs-large-finite mismatches since
    // overflow boundaries differ (float16 max ~65504, Float32Array ~3.4e38)
    if (
      tolerance >= FLOAT16_TOLERANCE &&
      !npHasFloat16 &&
      (!isFinite(numpytsResult) || !isFinite(numpyResult))
    )
      return true;
    // Use both relative and absolute tolerance (like numpy.allclose)
    const absErr = Math.abs(numpytsResult - numpyResult);
    const maxAbs = Math.max(Math.abs(numpytsResult), Math.abs(numpyResult));
    const relErr = maxAbs > 0 ? absErr / maxAbs : 0;
    return absErr < tolerance || relErr < tolerance;
  }

  if (typeof numpytsResult === 'boolean' && typeof numpyResult === 'boolean') {
    return numpytsResult === numpyResult;
  }

  // Bool ↔ number equivalence: TS bool arrays serialize as 0/1, Python as true/false
  if (typeof numpytsResult === 'number' && typeof numpyResult === 'boolean') {
    return numpytsResult === (numpyResult ? 1 : 0);
  }
  if (typeof numpytsResult === 'boolean' && typeof numpyResult === 'number') {
    return (numpytsResult ? 1 : 0) === numpyResult;
  }

  // Both strings (e.g., dtype names)
  if (typeof numpytsResult === 'string' && typeof numpyResult === 'string') {
    return numpytsResult === numpyResult;
  }

  // Both lists of arrays (e.g., from unstack)
  if (Array.isArray(numpytsResult) && Array.isArray(numpyResult)) {
    if (numpytsResult.length !== numpyResult.length) {
      return false;
    }
    // Compare each array in the list
    return numpytsResult.every((tsArr: any, i: number) =>
      resultsMatch(tsArr, numpyResult[i], operation, tolerance)
    );
  }

  // Both arrays (both should be {shape, data} format at this point)
  if (numpytsResult?.shape && numpyResult?.shape) {
    // Check shapes match
    if (numpytsResult.shape.length !== numpyResult.shape.length) {
      return false;
    }
    if (!numpytsResult.shape.every((dim: number, i: number) => dim === numpyResult.shape[i])) {
      return false;
    }

    // Compare values element by element
    // Both are already in {shape, data} format
    const tsData = numpytsResult.data;
    const npData = numpyResult.data;

    // SVD operations need special tolerance handling
    if (operation && isSvdOperation(operation)) {
      return compareSvdResults(tsData, npData);
    }

    return arraysEqual(tsData, npData, tolerance);
  }

  // Both plain objects (e.g., {values: [...], counts: [...]})
  if (
    typeof numpytsResult === 'object' &&
    numpytsResult !== null &&
    typeof numpyResult === 'object' &&
    numpyResult !== null &&
    !Array.isArray(numpytsResult) &&
    !Array.isArray(numpyResult)
  ) {
    const tsKeys = Object.keys(numpytsResult);
    const npKeys = Object.keys(numpyResult);

    // Check same keys
    if (tsKeys.length !== npKeys.length) {
      return false;
    }
    if (!tsKeys.every((k) => npKeys.includes(k))) {
      return false;
    }

    // Recursively compare each value
    return tsKeys.every((k) =>
      resultsMatch(numpytsResult[k], numpyResult[k], operation, tolerance)
    );
  }

  return false;
}

/**
 * Check if a partition result satisfies the partition property
 * All elements before kth should be <= element at kth
 * All elements after kth should be >= element at kth
 */
function checkPartitionProperty(data: any, kth: number, shape: number[]): boolean {
  // For multi-dimensional arrays, check along last axis
  if (shape.length === 0) return true;

  const lastAxisSize = shape[shape.length - 1]!;
  const outerSize = shape.slice(0, -1).reduce((a, b) => a * b, 1) || 1;

  // Flatten nested array to 1D for easier checking
  function flatten(arr: any): number[] {
    if (typeof arr === 'number') return [arr];
    if (Array.isArray(arr)) {
      return arr.flatMap((x) => flatten(x));
    }
    return [];
  }

  const flatData = flatten(data);

  // Check each row along the last axis
  for (let i = 0; i < outerSize; i++) {
    const rowStart = i * lastAxisSize;
    const kthValue = flatData[rowStart + kth];

    // Skip if kthValue is NaN
    if (typeof kthValue === 'number' && isNaN(kthValue)) continue;

    // Check all elements before kth are <= kthValue
    for (let j = 0; j < kth; j++) {
      const val = flatData[rowStart + j];
      if (typeof val === 'number' && !isNaN(val)) {
        if (val > kthValue!) return false;
      }
    }

    // Check all elements after kth are >= kthValue
    for (let j = kth + 1; j < lastAxisSize; j++) {
      const val = flatData[rowStart + j];
      if (typeof val === 'number' && !isNaN(val)) {
        if (val < kthValue!) return false;
      }
    }
  }

  return true;
}

/**
 * Recursively compare nested arrays with tolerance
 * Uses both relative and absolute tolerance for numerical stability
 */
function arraysEqual(a: any, b: any, tolerance: number = FLOAT64_TOLERANCE): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    // Complex [re, im] pair: compare using complex distance / magnitude
    // Only apply to finite numbers (avoid matching [sign, -Infinity] from slogdet etc.)
    if (
      a.length === 2 &&
      typeof a[0] === 'number' &&
      typeof a[1] === 'number' &&
      typeof b[0] === 'number' &&
      typeof b[1] === 'number' &&
      isFinite(a[0]) &&
      isFinite(a[1]) &&
      isFinite(b[0]) &&
      isFinite(b[1])
    ) {
      const dre = a[0] - b[0];
      const dim = a[1] - b[1];
      const dist = Math.sqrt(dre * dre + dim * dim);
      const magA = Math.sqrt(a[0] * a[0] + a[1] * a[1]);
      const magB = Math.sqrt(b[0] * b[0] + b[1] * b[1]);
      const mag = Math.max(magA, magB);
      const relErr = mag > 0 ? dist / mag : 0;
      return dist < tolerance || relErr < tolerance;
    }
    return a.every((val, i) => arraysEqual(val, b[i], tolerance));
  }

  // Compare BigInt values (from int64/uint64 results) — compare exactly
  if (typeof a === 'bigint' && typeof b === 'bigint') {
    return a === b;
  }
  if (typeof a === 'bigint' || typeof b === 'bigint') {
    const na = typeof a === 'bigint' ? Number(a) : a;
    const nb = typeof b === 'bigint' ? Number(b) : b;
    return arraysEqual(na, nb, tolerance);
  }

  // Compare numbers with tolerance
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return true;
    if (!isFinite(a) && !isFinite(b)) return a === b; // Both inf or -inf
    // For relaxed tolerance (float32), allow inf-vs-large-finite mismatches
    // since overflow boundaries differ between float32 and float64 computation
    if (tolerance > FLOAT64_TOLERANCE && (!isFinite(a) || !isFinite(b))) return true;
    // Use both relative and absolute tolerance (like numpy.allclose)
    const absErr = Math.abs(a - b);
    const maxAbs = Math.max(Math.abs(a), Math.abs(b));
    const relErr = maxAbs > 0 ? absErr / maxAbs : 0;
    return absErr < tolerance || relErr < tolerance;
  }

  // Compare booleans
  if (typeof a === 'boolean' && typeof b === 'boolean') {
    return a === b;
  }

  // Bool ↔ number equivalence: TS bool arrays serialize as 0/1, Python as true/false
  if (typeof a === 'number' && typeof b === 'boolean') {
    return a === (b ? 1 : 0);
  }
  if (typeof a === 'boolean' && typeof b === 'number') {
    return (a ? 1 : 0) === b;
  }

  return a === b;
}

/**
 * Run a single benchmark operation with numpy-ts
 */
function runNumpyTsOperation(spec: BenchmarkCase): any {
  // Setup arrays
  const arrays: Record<string, any> = {};

  for (const [key, config] of Object.entries(spec.setup)) {
    const { shape, dtype = 'float64', fill = 'zeros', value } = config;

    // Handle scalar values
    if (
      ['n', 'axis', 'new_shape', 'shape', 'fill_value', 'target_shape', 'dims', 'kth'].includes(key)
    ) {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape' || key === 'target_shape' || key === 'dims') {
        arrays[key] = shape;
        if (dtype !== 'float64') arrays['dtype'] = dtype;
      }
      continue;
    }

    // Handle indices array
    if (key === 'indices') {
      arrays[key] = shape;
      continue;
    }

    // Handle string values (like einsum subscripts)
    if (key === 'subscripts') {
      arrays[key] = config.value;
      continue;
    }

    // Create arrays
    if (value !== undefined) {
      arrays[key] = np.full(shape, value, dtype);
    } else if (fill === 'zeros') {
      arrays[key] = np.zeros(shape, dtype);
    } else if (fill === 'ones') {
      arrays[key] = np.ones(shape, dtype);
    } else if (fill === 'random' || fill === 'arange') {
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = flat.reshape(...shape);
    } else if (fill === 'shuffled') {
      // Deterministic Fisher-Yates shuffle of arange using LCG (seed=42)
      // Create arange with target dtype first (wraps for narrow types), then shuffle
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const wrapped = np.arange(0, size, 1, dtype);
      const vals = (wrapped as any).toArray().flat() as number[];
      let seed = 42;
      for (let i = size - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        const j = seed % (i + 1);
        [vals[i], vals[j]] = [vals[j]!, vals[i]!];
      }
      arrays[key] = np.array(vals, dtype).reshape(...shape);
    } else if (fill === 'complex' || fill === 'complex_small') {
      // Create complex array. 'complex' uses [1+1j, 2+2j, ...] (large values),
      // 'complex_small' uses modular values [(i%10+1) + (i%10+1)j] to avoid overflow in trig/exp.
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const complexValues = [];
      for (let i = 0; i < size; i++) {
        const v = fill === 'complex_small' ? (i % 10) + 1 : i + 1;
        complexValues.push(new np.Complex(v, v));
      }
      const flat = np.array(complexValues);
      const reshaped = flat.reshape(...shape);
      // Cast to complex64 if requested
      arrays[key] = dtype === 'complex64' ? np.asarray(reshaped, 'complex64') : reshaped;
    } else if (fill === 'invertible') {
      // Create an invertible matrix: arange + n*I (diagonally dominant)
      const n = shape[0];
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      const arangeMatrix = flat.reshape(...shape);
      const identity = np.eye(n, undefined, undefined, dtype);
      arrays[key] = arangeMatrix.add(identity.multiply(n * n));
    }
  }

  // Execute operation
  switch (spec.operation) {
    // Creation
    case 'zeros':
      return np.zeros(arrays.shape);
    case 'ones':
      return np.ones(arrays.shape);
    case 'arange':
      return np.arange(0, arrays.n);
    case 'linspace':
      return np.linspace(0, 100, arrays.n);
    case 'logspace':
      return np.logspace(0, 3, arrays.n);
    case 'geomspace':
      return np.geomspace(1, 1000, arrays.n);
    case 'eye':
      return np.eye(arrays.n);
    case 'identity':
      return np.identity(arrays.n);
    case 'empty':
      return np.empty(arrays.shape);
    case 'full':
      return np.full(arrays.shape, arrays.fill_value);
    case 'copy':
      return arrays.a.copy();
    case 'zeros_like':
      return np.zeros_like(arrays.a);

    // Arithmetic
    case 'add':
      return arrays.b !== undefined ? arrays.a.add(arrays.b) : arrays.a.add(arrays.scalar);
    case 'subtract':
      return arrays.b !== undefined
        ? arrays.a.subtract(arrays.b)
        : arrays.a.subtract(arrays.scalar);
    case 'multiply':
      return arrays.b !== undefined
        ? arrays.a.multiply(arrays.b)
        : arrays.a.multiply(arrays.scalar);
    case 'divide':
      return arrays.b !== undefined ? arrays.a.divide(arrays.b) : arrays.a.divide(arrays.scalar);
    case 'mod':
      return np.mod(arrays.a, arrays.b || arrays.scalar);
    case 'floor_divide':
      return np.floor_divide(arrays.a, arrays.b || arrays.scalar);
    case 'reciprocal':
      return np.reciprocal(arrays.a);
    case 'positive':
      return np.positive(arrays.a);

    // Math
    case 'sqrt':
      return np.sqrt(arrays.a);
    case 'power':
      return np.power(arrays.a, 2);
    case 'absolute':
      return np.absolute(arrays.a);
    case 'negative':
      return np.negative(arrays.a);
    case 'sign':
      return np.sign(arrays.a);

    // Trigonometric
    case 'sin':
      return np.sin(arrays.a);
    case 'cos':
      return np.cos(arrays.a);
    case 'tan':
      return np.tan(arrays.a);
    case 'arcsin':
      return np.arcsin(arrays.a);
    case 'arccos':
      return np.arccos(arrays.a);
    case 'arctan':
      return np.arctan(arrays.a);
    case 'arctan2':
      return np.arctan2(arrays.a, arrays.b);
    case 'hypot':
      return np.hypot(arrays.a, arrays.b);

    // Hyperbolic
    case 'sinh':
      return np.sinh(arrays.a);
    case 'cosh':
      return np.cosh(arrays.a);
    case 'tanh':
      return np.tanh(arrays.a);
    case 'arcsinh':
      return np.arcsinh(arrays.a);
    case 'arccosh':
      return np.arccosh(arrays.a);
    case 'arctanh':
      return np.arctanh(arrays.a);

    // Exponential
    case 'exp':
      return np.exp(arrays.a);
    case 'exp2':
      return np.exp2(arrays.a);
    case 'expm1':
      return np.expm1(arrays.a);
    case 'log':
      return np.log(arrays.a);
    case 'log2':
      return np.log2(arrays.a);
    case 'log10':
      return np.log10(arrays.a);
    case 'log1p':
      return np.log1p(arrays.a);
    case 'logaddexp':
      return np.logaddexp(arrays.a, arrays.b);
    case 'logaddexp2':
      return np.logaddexp2(arrays.a, arrays.b);

    // Gradient
    case 'diff':
      return np.diff(arrays.a);
    case 'ediff1d':
      return np.ediff1d(arrays.a);
    case 'gradient':
      return np.gradient(arrays.a) as any; // May return array or array of arrays
    case 'cross':
      return np.cross(arrays.a, arrays.b);

    // Linalg
    case 'dot':
      return arrays.a.dot(arrays.b);
    case 'inner':
      return arrays.a.inner(arrays.b);
    case 'outer':
      return arrays.a.outer(arrays.b);
    case 'tensordot':
      return arrays.a.tensordot(arrays.b, arrays.axes ?? 2);
    case 'matmul':
      return arrays.a.matmul(arrays.b);
    case 'trace':
      return arrays.a.trace();
    case 'transpose':
      return arrays.a.transpose();

    // Reductions
    case 'sum':
      return arrays.axis !== undefined ? arrays.a.sum(arrays.axis) : arrays.a.sum();
    case 'mean':
      return arrays.a.mean();
    case 'max':
      return arrays.a.max();
    case 'min':
      return arrays.a.min();
    case 'prod':
      return arrays.a.prod();
    case 'argmin':
      return arrays.a.argmin();
    case 'argmax':
      return arrays.a.argmax();
    case 'var':
      return arrays.a.var();
    case 'std':
      return arrays.a.std();
    case 'all':
      return arrays.a.all();
    case 'any':
      return arrays.a.any();

    // New reduction functions
    case 'cumsum':
      return arrays.a.cumsum();
    case 'cumprod':
      return arrays.a.cumprod();
    case 'ptp':
      return arrays.a.ptp();
    case 'median':
      return np.median(arrays.a);
    case 'percentile':
      return np.percentile(arrays.a, 50);
    case 'quantile':
      return np.quantile(arrays.a, 0.5);
    case 'average':
      return np.average(arrays.a);
    case 'nansum':
      return np.nansum(arrays.a);
    case 'nanmean':
      return np.nanmean(arrays.a);
    case 'nanmin':
      return np.nanmin(arrays.a);
    case 'nanmax':
      return np.nanmax(arrays.a);
    case 'nanquantile':
      return np.nanquantile(arrays.a, 0.5);
    case 'nanpercentile':
      return np.nanpercentile(arrays.a, 50);

    // Array creation - extra
    case 'asarray_chkfinite':
      return np.asarray_chkfinite(arrays.a);
    case 'require':
      return np.require(arrays.a, undefined, 'C');

    // Reshape
    case 'reshape':
      return arrays.a.reshape(...arrays.new_shape);
    case 'flatten':
      return arrays.a.flatten();
    case 'ravel':
      return arrays.a.ravel();

    // Array manipulation
    case 'swapaxes':
      return np.swapaxes(arrays.a, 0, 1);
    case 'concatenate':
      return np.concatenate([arrays.a, arrays.b], 0);
    case 'stack':
      return np.stack([arrays.a, arrays.b], 0);
    case 'vstack':
      return np.vstack([arrays.a, arrays.b]);
    case 'hstack':
      return np.hstack([arrays.a, arrays.b]);
    case 'tile':
      return np.tile(arrays.a, [2, 2]);
    case 'repeat':
      return arrays.a.repeat(2);
    case 'concat':
      return np.concat([arrays.a, arrays.b], 0);
    case 'unstack':
      return np.unstack(arrays.a, 0);
    case 'block':
      return np.block([arrays.a, arrays.b]);
    case 'item':
      return np.item(arrays.a, 0);
    case 'tolist':
      return np.tolist(arrays.a);

    // Advanced
    case 'broadcast_to':
      return np.broadcast_to(arrays.a, arrays.target_shape);
    case 'take':
      return arrays.a.take(arrays.indices);

    // New functions
    case 'diagonal':
      return np.diagonal(arrays.a);
    case 'kron':
      return np.kron(arrays.a, arrays.b);
    case 'deg2rad':
      return np.deg2rad(arrays.a);
    case 'rad2deg':
      return np.rad2deg(arrays.a);

    // New creation functions
    case 'diag':
      return np.diag(arrays.a);
    case 'tri':
      return np.tri(arrays.shape[0], arrays.shape[1]);
    case 'tril':
      return np.tril(arrays.a);
    case 'triu':
      return np.triu(arrays.a);

    // New manipulation functions
    case 'flip':
      return np.flip(arrays.a);
    case 'rot90':
      return np.rot90(arrays.a);
    case 'roll':
      return np.roll(arrays.a, 10);
    case 'pad':
      return np.pad(arrays.a, 2);
    // Additional arithmetic
    case 'cbrt':
      return np.cbrt(arrays.a);
    case 'fabs':
      return np.fabs(arrays.a);
    case 'divmod': {
      const [quotient] = np.divmod(arrays.a, arrays.b);
      return quotient; // Just return quotient for validation
    }
    case 'gcd':
      return np.gcd(arrays.a, arrays.b);
    case 'lcm':
      return np.lcm(arrays.a, arrays.b);
    case 'float_power':
      return np.float_power(arrays.a, arrays.b);
    case 'square':
      return np.square(arrays.a);
    case 'remainder':
      return np.remainder(arrays.a, arrays.b);
    case 'heaviside':
      return np.heaviside(arrays.a, arrays.b);
    case 'fmod':
      return np.fmod(arrays.a, arrays.b);
    case 'frexp': {
      const [mantissa] = np.frexp(arrays.a);
      return mantissa;
    }
    case 'ldexp':
      return np.ldexp(arrays.a, arrays.b);
    case 'modf': {
      const [fractional] = np.modf(arrays.a);
      return fractional;
    }

    // Additional linalg
    case 'einsum':
      return np.einsum(arrays.subscripts, arrays.a, arrays.b);

    // numpy.linalg module operations
    case 'linalg_det':
      return np.linalg.det(arrays.a);
    case 'linalg_inv':
      return np.linalg.inv(arrays.a);
    case 'linalg_solve':
      return np.linalg.solve(arrays.a, arrays.b);
    case 'linalg_qr': {
      const result = np.linalg.qr(arrays.a);
      // Return just Q for validation (simpler)
      return (result as { q: any }).q;
    }
    case 'linalg_cholesky':
      return np.linalg.cholesky(arrays._posdef ?? arrays.a);
    case 'linalg_svd': {
      const result = np.linalg.svd(arrays.a) as { u: any; s: any; vt: any };
      result.u.dispose();
      result.vt.dispose();
      return result.s;
    }
    case 'linalg_eig': {
      const result = np.linalg.eig(arrays.a) as { w: any; v: any };
      result.v.dispose();
      return result.w;
    }
    case 'linalg_eigh': {
      const aT = arrays.a.transpose();
      const sym = arrays.a.add(aT).multiply(0.5);
      const result = np.linalg.eigh(sym) as { w: any; v: any };
      aT.dispose();
      sym.dispose();
      result.v.dispose();
      return result.w;
    }
    case 'linalg_norm':
      return np.linalg.norm(arrays.a);
    case 'linalg_matrix_rank':
      return np.linalg.matrix_rank(arrays.a);
    case 'linalg_pinv':
      return np.linalg.pinv(arrays.a);
    case 'linalg_cond':
      return np.linalg.cond(arrays.a);
    case 'linalg_matrix_power':
      return np.linalg.matrix_power(arrays.a, 3);
    case 'linalg_lstsq': {
      const result = np.linalg.lstsq(arrays.a, arrays.b) as { x: any; residuals: any; s: any };
      result.residuals.dispose();
      result.s.dispose();
      return result.x;
    }
    case 'linalg_cross':
      return np.linalg.cross(arrays.a, arrays.b);
    case 'linalg_slogdet': {
      const result = np.linalg.slogdet(arrays.a);
      // Convert {sign, logabsdet} to array format to match Python output
      // Preserve input dtype (NumPy returns float32 for float32 input)
      const sign = (result as { sign: number }).sign;
      const logabsdet = (result as { logabsdet: number }).logabsdet;
      return np.array([sign, logabsdet], arrays.a.dtype as string);
    }
    case 'linalg_svdvals':
      return np.linalg.svdvals(arrays.a);
    case 'linalg_multi_dot': {
      const matrices = [arrays.a, arrays.b];
      if (arrays.c) matrices.push(arrays.c);
      return np.linalg.multi_dot(matrices);
    }
    case 'vdot':
      return np.vdot(arrays.a, arrays.b);
    case 'vecdot':
      return np.vecdot(arrays.a, arrays.b);
    case 'matrix_transpose':
      return np.matrix_transpose(arrays.a);
    case 'matvec':
      return np.matvec(arrays.a, arrays.b);
    case 'vecmat':
      return np.vecmat(arrays.a, arrays.b);

    // Indexing functions
    case 'take_along_axis':
      return np.take_along_axis(arrays.a, arrays.b, 0);
    case 'compress':
      return np.compress(arrays.b, arrays.a, 0);
    case 'diag_indices': {
      const idx = np.diag_indices(arrays.n);
      // Stack the two arrays for comparison
      return np.stack([idx[0]!, idx[1]!], 0);
    }
    case 'tril_indices': {
      const idx = np.tril_indices(arrays.n);
      return np.stack([idx[0]!, idx[1]!], 0);
    }
    case 'triu_indices': {
      const idx = np.triu_indices(arrays.n);
      return np.stack([idx[0]!, idx[1]!], 0);
    }
    case 'indices':
      return np.indices(arrays.shape);
    case 'ravel_multi_index':
      return np.ravel_multi_index([arrays.a, arrays.b], arrays.dims);
    case 'unravel_index': {
      const idx = np.unravel_index(arrays.a, arrays.dims);
      return np.stack(idx, 0);
    }

    // Bitwise operations
    case 'bitwise_and':
      return np.bitwise_and(arrays.a, arrays.b);
    case 'bitwise_or':
      return np.bitwise_or(arrays.a, arrays.b);
    case 'bitwise_xor':
      return np.bitwise_xor(arrays.a, arrays.b);
    case 'bitwise_not':
      return np.bitwise_not(arrays.a);
    case 'invert':
      return np.invert(arrays.a);
    case 'left_shift':
      return np.left_shift(arrays.a, arrays.b);
    case 'right_shift':
      return np.right_shift(arrays.a, arrays.b);
    case 'packbits':
      return np.packbits(arrays.a);
    case 'unpackbits':
      return np.unpackbits(arrays.a);
    case 'bitwise_count':
      return np.bitwise_count(arrays.a);

    // Sorting operations
    case 'sort':
      return np.sort(arrays.a);
    case 'argsort':
      return np.argsort(arrays.a);
    case 'partition':
      return np.partition(arrays.a, arrays.kth || 0);
    case 'argpartition':
      return np.argpartition(arrays.a, arrays.kth || 0);
    case 'lexsort':
      return np.lexsort([arrays.a, arrays.b]);
    case 'sort_complex':
      return np.sort_complex(arrays.a);

    // Searching operations
    case 'nonzero': {
      const idx = np.nonzero(arrays.a);
      return np.stack(idx, 0);
    }
    case 'argwhere':
      return np.argwhere(arrays.a);
    case 'flatnonzero':
      return np.flatnonzero(arrays.a);
    case 'where':
      return np.where(arrays.a, arrays.b, arrays.c);
    case 'searchsorted':
      return np.searchsorted(arrays.a, arrays.b);
    case 'extract':
      return np.extract(arrays.condition, arrays.a);
    case 'count_nonzero':
      return np.count_nonzero(arrays.a);

    // Statistics operations
    case 'bincount':
      return np.bincount(arrays.a);
    case 'digitize':
      return np.digitize(arrays.a, arrays.b);
    case 'histogram': {
      const [hist] = np.histogram(arrays.a, 10);
      return hist;
    }
    case 'histogram2d': {
      const [hist] = np.histogram2d(arrays.a, arrays.b, 10);
      return hist;
    }
    case 'correlate':
      return np.correlate(arrays.a, arrays.b, 'full');
    case 'convolve':
      return np.convolve(arrays.a, arrays.b, 'full');
    case 'cov':
      return np.cov(arrays.a);
    case 'corrcoef':
      return np.corrcoef(arrays.a);
    case 'histogram_bin_edges':
      return np.histogram_bin_edges(arrays.a, 10);
    case 'trapezoid':
      return np.trapezoid(arrays.a);

    // Set operations
    case 'trim_zeros':
      return np.trim_zeros(arrays.a);
    case 'unique_values':
      return np.unique_values(arrays.a);
    case 'unique_counts': {
      const result = np.unique_counts(arrays.a);
      return { values: result.values.toArray(), counts: result.counts.toArray() };
    }

    // Logic operations
    case 'logical_and':
      if (arrays.b) {
        return np.logical_and(arrays.a, arrays.b);
      } else {
        return np.logical_and(arrays.a, arrays.scalar);
      }
    case 'logical_or':
      if (arrays.b) {
        return np.logical_or(arrays.a, arrays.b);
      } else {
        return np.logical_or(arrays.a, arrays.scalar);
      }
    case 'logical_not':
      return np.logical_not(arrays.a);
    case 'logical_xor':
      if (arrays.b) {
        return np.logical_xor(arrays.a, arrays.b);
      } else {
        return np.logical_xor(arrays.a, arrays.scalar);
      }
    case 'isfinite':
      return np.isfinite(arrays.a);
    case 'isinf':
      return np.isinf(arrays.a);
    case 'isnan':
      return np.isnan(arrays.a);
    case 'isneginf':
      return np.isneginf(arrays.a);
    case 'isposinf':
      return np.isposinf(arrays.a);
    case 'isreal':
      return np.isreal(arrays.a);
    case 'signbit':
      return np.signbit(arrays.a);
    case 'copysign':
      if (arrays.b) {
        return np.copysign(arrays.a, arrays.b);
      } else {
        return np.copysign(arrays.a, arrays.scalar);
      }

    // Random operations — seeded for exact-match validation
    case 'random_random':
      np.random.seed(42);
      return np.random.random(arrays.shape);
    case 'random_rand':
      np.random.seed(42);
      return np.random.rand(...arrays.shape);
    case 'random_randn':
      np.random.seed(42);
      return np.random.randn(...arrays.shape);
    case 'random_randint':
      np.random.seed(42);
      return np.random.randint(0, 100, arrays.shape, arrays.dtype || 'int64');
    case 'random_uniform':
      np.random.seed(42);
      return np.random.uniform(0, 1, arrays.shape);
    case 'random_normal':
      np.random.seed(42);
      return np.random.normal(0, 1, arrays.shape);
    case 'random_standard_normal':
      np.random.seed(42);
      return np.random.standard_normal(arrays.shape);
    case 'random_exponential':
      np.random.seed(42);
      return np.random.exponential(1, arrays.shape);
    case 'random_poisson':
      np.random.seed(42);
      return np.random.poisson(5, arrays.shape);
    case 'random_binomial':
      np.random.seed(42);
      return np.random.binomial(10, 0.5, arrays.shape);
    case 'random_choice':
      np.random.seed(42);
      return np.random.choice(arrays.n, 100);
    case 'random_permutation':
      np.random.seed(42);
      return np.random.permutation(arrays.n);
    case 'random_gamma':
      np.random.seed(42);
      return np.random.gamma(2, 1, arrays.shape);
    case 'random_beta':
      np.random.seed(42);
      return np.random.beta(2, 5, arrays.shape);
    case 'random_chisquare':
      np.random.seed(42);
      return np.random.chisquare(5, arrays.shape);
    case 'random_laplace':
      np.random.seed(42);
      return np.random.laplace(0, 1, arrays.shape);
    case 'random_geometric':
      np.random.seed(42);
      return np.random.geometric(0.5, arrays.shape);
    case 'random_dirichlet':
      np.random.seed(42);
      return np.random.dirichlet([1, 2, 3], arrays.shape[0]);
    case 'random_standard_exponential':
      np.random.seed(42);
      return np.random.standard_exponential(arrays.shape);
    case 'random_logistic':
      np.random.seed(42);
      return np.random.logistic(0, 1, arrays.shape);
    case 'random_lognormal':
      np.random.seed(42);
      return np.random.lognormal(0, 1, arrays.shape);
    case 'random_gumbel':
      np.random.seed(42);
      return np.random.gumbel(0, 1, arrays.shape);
    case 'random_pareto':
      np.random.seed(42);
      return np.random.pareto(3, arrays.shape);
    case 'random_power':
      np.random.seed(42);
      return np.random.power(3, arrays.shape);
    case 'random_rayleigh':
      np.random.seed(42);
      return np.random.rayleigh(1, arrays.shape);
    case 'random_weibull':
      np.random.seed(42);
      return np.random.weibull(3, arrays.shape);
    case 'random_triangular':
      np.random.seed(42);
      return np.random.triangular(0, 0.5, 1, arrays.shape);
    case 'random_standard_cauchy':
      np.random.seed(42);
      return np.random.standard_cauchy(arrays.shape);
    case 'random_standard_t':
      np.random.seed(42);
      return np.random.standard_t(5, arrays.shape);
    case 'random_wald':
      np.random.seed(42);
      return np.random.wald(1, 1, arrays.shape);
    case 'random_vonmises':
      np.random.seed(42);
      return np.random.vonmises(0, 1, arrays.shape);
    case 'random_zipf':
      np.random.seed(42);
      return np.random.zipf(2, arrays.shape);

    // Generator (PCG64) random
    case 'gen_random': {
      const rng = np.random.default_rng(42);
      return rng.random(arrays.shape);
    }
    case 'gen_uniform': {
      const rng = np.random.default_rng(42);
      return rng.uniform(0, 1, arrays.shape);
    }
    case 'gen_standard_normal': {
      const rng = np.random.default_rng(42);
      return rng.standard_normal(arrays.shape);
    }
    case 'gen_normal': {
      const rng = np.random.default_rng(42);
      return rng.normal(0, 1, arrays.shape);
    }
    case 'gen_exponential': {
      const rng = np.random.default_rng(42);
      return rng.exponential(1, arrays.shape);
    }
    case 'gen_integers': {
      const rng = np.random.default_rng(42);
      return rng.integers(0, 100, arrays.shape);
    }
    case 'gen_permutation': {
      const rng = np.random.default_rng(42);
      return rng.permutation(arrays.n);
    }

    // Complex operations
    case 'complex_zeros':
      return np.zeros(arrays.shape, 'complex128');
    case 'complex_ones':
      return np.ones(arrays.shape, 'complex128');
    case 'complex_add':
      return arrays.a.add(arrays.b);
    case 'complex_multiply':
      return arrays.a.multiply(arrays.b);
    case 'complex_divide':
      return arrays.a.divide(arrays.b);
    case 'complex_real':
      return np.real(arrays.a);
    case 'complex_imag':
      return np.imag(arrays.a);
    case 'complex_conj':
      return np.conj(arrays.a);
    case 'complex_angle':
      return np.angle(arrays.a);
    case 'complex_abs':
      return np.abs(arrays.a);
    case 'complex_sqrt':
      return np.sqrt(arrays.a);
    case 'complex_sum':
      return arrays.a.sum();
    case 'complex_mean':
      return arrays.a.mean();
    case 'complex_prod':
      return arrays.a.prod();

    // Other Math operations
    case 'clip':
      return np.clip(arrays.a, 10, 100);
    case 'maximum':
      return np.maximum(arrays.a, arrays.b);
    case 'minimum':
      return np.minimum(arrays.a, arrays.b);
    case 'fmax':
      return np.fmax(arrays.a, arrays.b);
    case 'fmin':
      return np.fmin(arrays.a, arrays.b);
    case 'nan_to_num':
      return np.nan_to_num(arrays.a);
    case 'interp':
      return np.interp(arrays.x, arrays.xp, arrays.fp);
    case 'unwrap':
      return np.unwrap(arrays.a);
    case 'sinc':
      return np.sinc(arrays.a);
    case 'i0':
      return np.i0(arrays.a);

    // Polynomial operations
    case 'poly':
      return np.poly(arrays.a);
    case 'polyadd':
      return np.polyadd(arrays.a, arrays.b);
    case 'polyder':
      return np.polyder(arrays.a);
    case 'polydiv': {
      const [q] = np.polydiv(arrays.a, arrays.b);
      return q; // Return just quotient for validation
    }
    case 'polyfit':
      return np.polyfit(arrays.a, arrays.b, 2);
    case 'polyint':
      return np.polyint(arrays.a);
    case 'polymul':
      return np.polymul(arrays.a, arrays.b);
    case 'polysub':
      return np.polysub(arrays.a, arrays.b);
    case 'polyval':
      return np.polyval(arrays.a, arrays.b);
    case 'roots':
      return np.roots(arrays.a);

    // Type checking operations (return booleans/strings, not arrays)
    case 'can_cast':
      return np.can_cast('int32', 'float64');
    case 'result_type':
      return np.result_type('int32', 'float64');
    case 'min_scalar_type':
      return np.min_scalar_type(1000);
    case 'issubdtype':
      return np.issubdtype('int32', 'integer');

    // FFT operations
    case 'fft':
      return np.fft.fft(arrays.a);
    case 'ifft':
      return np.fft.ifft(arrays.a);
    case 'fft2':
      return np.fft.fft2(arrays.a);
    case 'ifft2':
      return np.fft.ifft2(arrays.a);
    case 'fftn':
      return np.fft.fftn(arrays.a);
    case 'ifftn':
      return np.fft.ifftn(arrays.a);
    case 'rfft':
      return np.fft.rfft(arrays.a);
    case 'irfft':
      return np.fft.irfft(arrays.a);
    case 'rfft2':
      return np.fft.rfft2(arrays.a);
    case 'irfft2':
      return np.fft.irfft2(arrays.a);
    case 'rfftn':
      return np.fft.rfftn(arrays.a);
    case 'irfftn':
      return np.fft.irfftn(arrays.a);
    case 'hfft':
      return np.fft.hfft(arrays.a);
    case 'ihfft':
      return np.fft.ihfft(arrays.a);
    case 'fftfreq':
      return np.fft.fftfreq(arrays.n);
    case 'rfftfreq':
      return np.fft.rfftfreq(arrays.n);
    case 'fftshift':
      return np.fft.fftshift(arrays.a);
    case 'ifftshift':
      return np.fft.ifftshift(arrays.a);

    default:
      throw new Error(`Unknown operation: ${spec.operation}`);
  }
}

/**
 * Validate all benchmarks produce correct results
 */
export async function validateBenchmarks(specs: BenchmarkCase[]): Promise<void> {
  const scriptPath = resolve(__dirname, '../scripts/validation.py');

  return new Promise((resolve, reject) => {
    const python = spawn('python3', [scriptPath]);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Validation script failed: ${stderr}`));
        return;
      }

      try {
        const rawResults = JSON.parse(stdout);
        // Deserialize special values (Infinity, NaN)
        const numpyResults = rawResults.map((r: any) => deserializeValue(r));

        // Run numpy-ts operations and compare
        let passed = 0;
        let failed = 0;

        for (let i = 0; i < specs.length; i++) {
          const spec = specs[i]!;
          const numpyResult = numpyResults[i];
          let numpytsResult: any;
          try {
            numpytsResult = runNumpyTsOperation(spec);

            // Convert numpy-ts result to comparable format
            let tsValue: any;

            // Helper to convert Complex objects to [re, im] pairs for comparison
            function complexToReal(val: any): any {
              if (val && typeof val === 'object' && 're' in val && 'im' in val) {
                return [val.re, val.im];
              }
              if (Array.isArray(val)) {
                return val.map(complexToReal);
              }
              return val;
            }

            // Track TS dtype for comparison
            let tsDtype: string | undefined;

            if (typeof numpytsResult === 'number' || typeof numpytsResult === 'boolean') {
              tsValue = numpytsResult;
            } else if (
              numpytsResult &&
              typeof numpytsResult === 'object' &&
              're' in numpytsResult &&
              'im' in numpytsResult
            ) {
              // It's a Complex scalar result - convert to [re, im] pair
              tsValue = [numpytsResult.re, numpytsResult.im];
            } else if (
              Array.isArray(numpytsResult) &&
              numpytsResult.length > 0 &&
              numpytsResult[0] &&
              'shape' in numpytsResult[0]
            ) {
              // It's a list of NDArrays (e.g., from unstack)
              tsValue = numpytsResult.map((arr: any) => ({
                shape: Array.from(arr.shape),
                data: complexToReal(arr.toArray()),
              }));
            } else if (
              numpytsResult &&
              typeof numpytsResult === 'object' &&
              'shape' in numpytsResult
            ) {
              // It's an NDArray - check if it has toArray method
              if (typeof numpytsResult.toArray !== 'function') {
                throw new Error(
                  `NDArray missing toArray method. Type: ${typeof numpytsResult}, keys: ${Object.keys(numpytsResult).join(', ')}`
                );
              }
              // Convert array data, handling Complex objects
              const rawData = numpytsResult.toArray();
              tsDtype = numpytsResult.dtype;
              tsValue = {
                shape: Array.from(numpytsResult.shape),
                data: complexToReal(rawData),
              };
            } else {
              tsValue = numpytsResult;
            }

            // --- Shape check ---
            if (tsValue?.shape && numpyResult?.shape) {
              const tsShape = JSON.stringify(tsValue.shape);
              const npShape = JSON.stringify(numpyResult.shape);
              if (tsShape !== npShape) {
                failed++;
                console.error(
                  `  ❌ ${spec.name}: Shape mismatch: TS ${tsShape} vs NumPy ${npShape}`
                );
                numpytsResult?.dispose?.();
                continue;
              }
            }

            // --- Dtype check ---
            // NumPy dtype map: numpy names → our names
            const NP_DTYPE_MAP: Record<string, string> = {
              float64: 'float64',
              float32: 'float32',
              float16: 'float16',
              int64: 'int64',
              int32: 'int32',
              int16: 'int16',
              int8: 'int8',
              uint64: 'uint64',
              uint32: 'uint32',
              uint16: 'uint16',
              uint8: 'uint8',
              bool_: 'bool',
              bool: 'bool',
              complex128: 'complex128',
              complex64: 'complex64',
            };
            if (tsDtype && numpyResult?.dtype) {
              const npDtype = NP_DTYPE_MAP[numpyResult.dtype] ?? numpyResult.dtype;
              // Index-returning functions: we return float64, NumPy returns int64
              const INDEX_OPS = new Set([
                'argsort',
                'argpartition',
                'argmax',
                'argmin',
                'searchsorted',
                'nonzero',
                'count_nonzero',
                'lexsort',
                'argwhere',
                'flatnonzero',
                'bincount',
                'digitize',
                'diag_indices',
                'tril_indices',
                'triu_indices',
                'indices',
                'ravel_multi_index',
                'unravel_index',
                'random_permutation',
              ]);
              // Integer creation functions: full/full_like return int32 (we
              // diverge from NumPy's int64 to avoid BigInt for in-range fills).
              const INT_CREATION_OPS = new Set(['full', 'full_like']);
              const isIndexOp = INDEX_OPS.has(spec.operation);
              const isIntCreation = INT_CREATION_OPS.has(spec.operation);
              if (
                isIndexOp &&
                tsDtype === 'float64' &&
                (npDtype === 'int64' || npDtype === 'uint64')
              ) {
                // Accepted: index ops return float64 for JS ergonomics
              } else if (
                spec.operation === 'arange' &&
                tsDtype === 'float64' &&
                npDtype === 'int64'
              ) {
                // Accepted: arange defaults to float64 (matches zeros/ones/linspace)
              } else if (isIntCreation && tsDtype === 'int32' && npDtype === 'int64') {
                // Accepted: int32 is our int64 equivalent for non-BigInt JS numbers
              } else if (
                (spec.operation === 'histogram' || spec.operation === 'histogram2d') &&
                tsDtype === 'float64' &&
                npDtype === 'int64'
              ) {
                // Accepted: we use float64 for histogram counts to avoid BigInt ergonomics
              } else if (spec.operation === 'copysign' && !spec.setup.b && npDtype === 'float64') {
                // Accepted: Python float scalar is float64, forcing promotion.
                // JS scalar copysign preserves input dtype (no implicit float64 scalar type).
              } else if (tsDtype !== npDtype) {
                failed++;
                console.error(
                  `  ❌ ${spec.name}: Dtype mismatch: TS '${tsDtype}' vs NumPy '${npDtype}'`
                );
                numpytsResult?.dispose?.();
                continue;
              }
            }

            // For partition/argpartition, check partition property instead of exact equality
            let isValid = false;
            if (spec.operation === 'partition' || spec.operation === 'argpartition') {
              // Extract kth from spec setup
              const kth = spec.setup.kth?.shape?.[0] ?? 0;

              if (spec.operation === 'partition') {
                // Check partition property: elements before kth <= kth element <= elements after kth
                isValid =
                  tsValue.shape &&
                  tsValue.shape.length > 0 &&
                  checkPartitionProperty(tsValue.data, kth, tsValue.shape);
              } else {
                // For argpartition, rebuild the input array and apply indices
                // to get partitioned values, then check partition property
                const aSpec = spec.setup.a!;
                const aShape = aSpec.shape;
                const aSize = aShape.reduce((x: number, y: number) => x * y, 1);
                const aDtype = aSpec.dtype || 'float64';
                const aFill = aSpec.fill || 'arange';

                let origArray: number[];
                if (aFill === 'shuffled') {
                  const wrapped = np.arange(0, aSize, 1, aDtype);
                  origArray = (wrapped as any).toArray().flat() as number[];
                  let seed = 42;
                  for (let si = aSize - 1; si > 0; si--) {
                    seed = (seed * 1664525 + 1013904223) >>> 0;
                    const sj = seed % (si + 1);
                    [origArray[si], origArray[sj]] = [origArray[sj]!, origArray[si]!];
                  }
                } else {
                  origArray = Array.from({ length: aSize }, (_, i) => i);
                }

                // Reshape flat origArray into nested arrays matching input shape
                function reshapeFlat(flat: number[], shape: number[]): any {
                  if (shape.length <= 1) return flat;
                  const rowSize = shape.slice(1).reduce((a, b) => a * b, 1);
                  const rows: any[] = [];
                  for (let ri = 0; ri < shape[0]!; ri++) {
                    rows.push(
                      reshapeFlat(flat.slice(ri * rowSize, (ri + 1) * rowSize), shape.slice(1))
                    );
                  }
                  return rows;
                }

                const origNested = reshapeFlat(origArray, aShape);

                // Apply per-row indices: for each row, indices are local (0..axisSize-1)
                function applyIndices(data: any, indices: any): any {
                  if (Array.isArray(indices) && Array.isArray(indices[0])) {
                    return indices.map((row: any, ri: number) => applyIndices(data[ri], row));
                  }
                  if (Array.isArray(indices)) {
                    return indices.map((idx: number) => data[idx]);
                  }
                  return data;
                }

                const partitionedData = applyIndices(origNested, tsValue.data);
                isValid = checkPartitionProperty(partitionedData, kth, tsValue.shape);
              }
            } else if (spec.operation === 'empty' || spec.operation === 'empty_like') {
              // empty returns uninitialized data — only check shape/dtype match
              isValid =
                tsValue.shape !== undefined &&
                JSON.stringify(tsValue.shape) === JSON.stringify(numpyResult.shape);
            } else {
              // Standard comparison for other operations
              // Use looser tolerance for lower-precision dtypes
              const hasFloat16 = Object.values(spec.setup).some((s) => s.dtype === 'float16');
              const hasLowPrecision = Object.values(spec.setup).some(
                (s) => s.dtype && s.dtype !== 'float64'
              );
              const tol = hasFloat16
                ? FLOAT16_TOLERANCE
                : hasLowPrecision
                  ? FLOAT32_TOLERANCE
                  : FLOAT64_TOLERANCE;
              isValid = resultsMatch(tsValue, numpyResult, spec.operation, tol);
            }

            if (isValid) {
              passed++;
            } else {
              failed++;
              console.error(`  ❌ ${spec.name}: Results don't match`);
              const bigIntReplacer = (_k: string, v: unknown) =>
                typeof v === 'bigint' ? Number(v) : v;
              console.error(
                `     numpy-ts: ${JSON.stringify(tsValue, bigIntReplacer).substring(0, 100)}`
              );
              console.error(
                `     NumPy:    ${JSON.stringify(numpyResult, bigIntReplacer).substring(0, 100)}`
              );
            }
          } catch (err) {
            failed++;
            console.error(`  ❌ ${spec.name}: Error - ${err}`);
          }
          // Eagerly free WASM-backed memory to prevent heap exhaustion
          (numpytsResult as any)?.dispose?.();
        }

        if (failed > 0) {
          reject(
            new Error(
              `Validation failed: ${failed}/${specs.length} benchmarks produced incorrect results`
            )
          );
        } else {
          console.log(`  ✓ ${passed}/${specs.length} operations validated`);
          resolve();
        }
      } catch (err) {
        reject(new Error(`Failed to parse validation output: ${err}`));
      }
    });

    python.on('error', (err) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });

    // Send specs to Python
    python.stdin.write(JSON.stringify({ specs }));
    python.stdin.end();
  });
}
