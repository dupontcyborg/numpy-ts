/**
 * Validation module for benchmark correctness
 * Validates that numpy-ts produces the same results as NumPy
 */

import { spawn } from 'child_process';
import { resolve } from 'path';
import * as np from '../../src/index';
import type { BenchmarkCase } from './types';

const FLOAT_TOLERANCE = 1e-10;

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
  }
  return val;
}

/**
 * Check if operation is a random operation (non-deterministic)
 */
function isRandomOperation(operation: string): boolean {
  return operation.startsWith('random_');
}

/**
 * Compare two arrays or scalars for equality with tolerance
 * @param operation - The operation name (for special handling of non-deterministic operations)
 */
function resultsMatch(numpytsResult: any, numpyResult: any, operation?: string): boolean {
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

  // Both null (often represents NaN or Infinity in serialization)
  if (numpytsResult === null && numpyResult === null) {
    return true;
  }

  // Both scalars
  if (typeof numpytsResult === 'number' && typeof numpyResult === 'number') {
    // Handle NaN: both NaN is considered equal
    if (isNaN(numpytsResult) && isNaN(numpyResult)) return true;
    // Handle Infinity: both must be same infinity
    if (!isFinite(numpytsResult) && !isFinite(numpyResult)) return numpytsResult === numpyResult;
    return Math.abs(numpytsResult - numpyResult) < FLOAT_TOLERANCE;
  }

  if (typeof numpytsResult === 'boolean' && typeof numpyResult === 'boolean') {
    return numpytsResult === numpyResult;
  }

  // Both lists of arrays (e.g., from unstack)
  if (Array.isArray(numpytsResult) && Array.isArray(numpyResult)) {
    if (numpytsResult.length !== numpyResult.length) {
      return false;
    }
    // Compare each array in the list
    return numpytsResult.every((tsArr: any, i: number) => resultsMatch(tsArr, numpyResult[i], operation));
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

    return arraysEqual(tsData, npData);
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
    return tsKeys.every((k) => resultsMatch(numpytsResult[k], numpyResult[k], operation));
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
 */
function arraysEqual(a: any, b: any): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, i) => arraysEqual(val, b[i]));
  }

  // Compare numbers with tolerance
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return true;
    if (!isFinite(a) && !isFinite(b)) return a === b; // Both inf or -inf
    return Math.abs(a - b) < FLOAT_TOLERANCE;
  }

  // Compare booleans
  if (typeof a === 'boolean' && typeof b === 'boolean') {
    return a === b;
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
    if (['n', 'axis', 'new_shape', 'shape', 'fill_value', 'target_shape', 'dims', 'kth'].includes(key)) {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape' || key === 'target_shape' || key === 'dims') {
        arrays[key] = shape;
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
    } else if (fill === 'complex') {
      // Create complex array with [1+1j, 2+2j, 3+3j, ...]
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const complexValues = [];
      for (let i = 0; i < size; i++) {
        complexValues.push(new np.Complex(i + 1, i + 1));
      }
      const flat = np.array(complexValues);
      arrays[key] = flat.reshape(...shape);
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
    case 'multiply':
      return arrays.b !== undefined
        ? arrays.a.multiply(arrays.b)
        : arrays.a.multiply(arrays.scalar);
    case 'mod':
      return np.mod(arrays.a, arrays.b || arrays.scalar);
    case 'floor_divide':
      return np.floor_divide(arrays.a, arrays.b || arrays.scalar);
    case 'reciprocal':
      return np.reciprocal(arrays.a);

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
    case 'linalg_cholesky': {
      // Create positive definite matrix: A^T * A + n*I
      const n = arrays.a.shape[0];
      const aT = arrays.a.transpose();
      const aTa = aT.matmul(arrays.a);
      const posdef = aTa.add(np.eye(n).multiply(n));
      return np.linalg.cholesky(posdef);
    }
    case 'linalg_svd': {
      const result = np.linalg.svd(arrays.a);
      // Return just S (singular values) for validation
      return (result as { s: any }).s;
    }
    case 'linalg_eig': {
      const result = np.linalg.eig(arrays.a);
      // Return just eigenvalues for validation (eigenvectors can differ)
      return (result as { w: any }).w;
    }
    case 'linalg_eigh': {
      // Create symmetric matrix: (A + A^T) / 2
      const aT = arrays.a.transpose();
      const sym = arrays.a.add(aT).multiply(0.5);
      const result = np.linalg.eigh(sym);
      // Return just eigenvalues for validation
      return (result as { w: any }).w;
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
      const result = np.linalg.lstsq(arrays.a, arrays.b);
      // Return just x (solution) for validation
      return (result as { x: any }).x;
    }
    case 'linalg_cross':
      return np.linalg.cross(arrays.a, arrays.b);

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

    // Random operations - these are non-deterministic, so we just validate they run
    case 'random_random':
      return np.random.random(arrays.shape);
    case 'random_rand':
      return np.random.rand(...arrays.shape);
    case 'random_randn':
      return np.random.randn(...arrays.shape);
    case 'random_randint':
      return np.random.randint(0, 100, arrays.shape);
    case 'random_uniform':
      return np.random.uniform(0, 1, arrays.shape);
    case 'random_normal':
      return np.random.normal(0, 1, arrays.shape);
    case 'random_standard_normal':
      return np.random.standard_normal(arrays.shape);
    case 'random_exponential':
      return np.random.exponential(1, arrays.shape);
    case 'random_poisson':
      return np.random.poisson(5, arrays.shape);
    case 'random_binomial':
      return np.random.binomial(10, 0.5, arrays.shape);
    case 'random_choice':
      return np.random.choice(arrays.n, 100);
    case 'random_permutation':
      return np.random.permutation(arrays.n);

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

          try {
            const numpytsResult = runNumpyTsOperation(spec);

            // Convert numpy-ts result to comparable format
            let tsValue: any;

            // Helper to convert Complex objects to real parts
            function complexToReal(val: any): any {
              if (val && typeof val === 'object' && 're' in val && 'im' in val) {
                // It's a Complex object - extract real part
                return val.re;
              }
              if (Array.isArray(val)) {
                return val.map(complexToReal);
              }
              return val;
            }

            if (typeof numpytsResult === 'number' || typeof numpytsResult === 'boolean') {
              tsValue = numpytsResult;
            } else if (numpytsResult && typeof numpytsResult === 'object' && 're' in numpytsResult && 'im' in numpytsResult) {
              // It's a Complex scalar result - convert to real
              tsValue = numpytsResult.re;
            } else if (Array.isArray(numpytsResult) && numpytsResult.length > 0 && numpytsResult[0] && 'shape' in numpytsResult[0]) {
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
              tsValue = {
                shape: Array.from(numpytsResult.shape),
                data: complexToReal(rawData),
              };
            } else {
              tsValue = numpytsResult;
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
                // For argpartition, apply indices to get partitioned values and check property
                // Get original array from spec setup
                const origArray = Array.isArray(spec.setup.a)
                  ? spec.setup.a
                  : spec.setup.a?.shape
                    ? Array.from({ length: spec.setup.a.shape.reduce((a: number, b: number) => a * b, 1) }, (_, i) => i)
                    : [];

                // Apply indices to get partitioned array
                function applyIndices(data: any, indices: any): any {
                  if (Array.isArray(indices)) {
                    return indices.map((idx: any) => {
                      if (Array.isArray(idx)) {
                        return applyIndices(data, idx);
                      }
                      return Array.isArray(data) ? data[idx] : data;
                    });
                  }
                  return data;
                }

                const partitionedData = applyIndices(origArray, tsValue.data);
                isValid = checkPartitionProperty(partitionedData, kth, tsValue.shape);
              }
            } else {
              // Standard comparison for other operations
              isValid = resultsMatch(tsValue, numpyResult, spec.operation);
            }

            if (isValid) {
              passed++;
            } else {
              failed++;
              console.error(`  ❌ ${spec.name}: Results don't match`);
              console.error(`     numpy-ts: ${JSON.stringify(tsValue).substring(0, 100)}`);
              console.error(`     NumPy:    ${JSON.stringify(numpyResult).substring(0, 100)}`);
            }
          } catch (err) {
            failed++;
            console.error(`  ❌ ${spec.name}: Error - ${err}`);
          }
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
