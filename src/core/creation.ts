/**
 * Array Creation Functions - Tree-shakeable standalone module
 *
 * This module provides all array creation functions including:
 * - Basic: zeros, ones, empty, full, array, arange, linspace, etc.
 * - Advanced: diag, tri, meshgrid, vander, etc.
 *
 * All functions return NDArrayCore for optimal tree-shaking.
 */

import { ArrayStorage } from '../common/storage';
import { NDArrayCore, type DType } from '../common/ndarray-core';
import { Complex, isComplexLike } from '../common/complex';
import {
  getTypedArrayConstructor,
  isBigIntDType,
  isComplexDType,
  DEFAULT_DTYPE,
  type TypedArray,
} from '../common/dtype';

// Re-export types
export type { DType, TypedArray } from '../common/dtype';

// Helper to convert ArrayStorage to NDArrayCore
function fromStorage(storage: ArrayStorage): NDArrayCore {
  return new NDArrayCore(storage);
}

// Helper to check if input is an NDArrayCore or compatible object (like NDArray)
// This handles the case where NDArray doesn't extend NDArrayCore
function isNDArrayLike(a: unknown): a is NDArrayCore {
  if (a instanceof NDArrayCore) return true;
  // Also accept objects with storage property (like NDArray from full/)
  if (
    a &&
    typeof a === 'object' &&
    'storage' in a &&
    (a as { storage: unknown }).storage instanceof ArrayStorage
  ) {
    return true;
  }
  return false;
}

// ============================================================
// Basic Array Creation
// ============================================================

/**
 * Create array of zeros
 */
export function zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArrayCore {
  const storage = ArrayStorage.zeros(shape, dtype);
  return new NDArrayCore(storage);
}

/**
 * Create array of ones
 */
export function ones(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArrayCore {
  const storage = ArrayStorage.ones(shape, dtype);
  return new NDArrayCore(storage);
}

/**
 * Create an uninitialized array (zeros in JS)
 */
export function empty(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArrayCore {
  return zeros(shape, dtype);
}

/**
 * Create array filled with a constant value
 */
export function full(
  shape: number[],
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArrayCore {
  let actualDtype = dtype;
  if (!actualDtype) {
    if (typeof fill_value === 'bigint') {
      actualDtype = 'int64';
    } else if (typeof fill_value === 'boolean') {
      actualDtype = 'bool';
    } else if (Number.isInteger(fill_value)) {
      actualDtype = 'int32';
    } else {
      actualDtype = DEFAULT_DTYPE;
    }
  }

  const Constructor = getTypedArrayConstructor(actualDtype);
  if (!Constructor) {
    throw new Error(`Cannot create full array with dtype ${actualDtype}`);
  }
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Constructor(size);

  if (isBigIntDType(actualDtype)) {
    const bigintValue =
      typeof fill_value === 'bigint' ? fill_value : BigInt(Math.round(Number(fill_value)));
    (data as BigInt64Array | BigUint64Array).fill(bigintValue);
  } else if (actualDtype === 'bool') {
    (data as Uint8Array).fill(fill_value ? 1 : 0);
  } else {
    (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(Number(fill_value));
  }

  const storage = ArrayStorage.fromData(data, shape, actualDtype);
  return new NDArrayCore(storage);
}

// Helper functions for array()
function inferShape(data: unknown): number[] {
  const shape: number[] = [];
  let current = data;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }
  return shape;
}

function containsBigInt(data: unknown): boolean {
  if (typeof data === 'bigint') return true;
  if (Array.isArray(data)) {
    return data.some((item) => containsBigInt(item));
  }
  return false;
}

function containsComplex(data: unknown): boolean {
  if (isComplexLike(data)) return true;
  if (Array.isArray(data)) {
    return data.some((item) => containsComplex(item));
  }
  return false;
}

function flattenKeepBigInt(data: unknown): unknown[] {
  const result: unknown[] = [];
  function flatten(arr: unknown): void {
    if (Array.isArray(arr)) {
      arr.forEach((item) => flatten(item));
    } else {
      result.push(arr);
    }
  }
  flatten(data);
  return result;
}

/**
 * Create array from nested JavaScript arrays
 */
export function array(data: unknown, dtype?: DType): NDArrayCore {
  if (data instanceof NDArrayCore) {
    if (!dtype || data.dtype === dtype) {
      return data.copy();
    }
    return data.astype(dtype);
  }

  const hasBigInt = containsBigInt(data);
  const hasComplex = containsComplex(data);

  const shape = inferShape(data);
  const size = shape.reduce((a: number, b: number) => a * b, 1);

  let actualDtype = dtype;
  if (!actualDtype) {
    if (hasComplex) {
      actualDtype = 'complex128';
    } else if (hasBigInt) {
      actualDtype = 'int64';
    } else {
      actualDtype = DEFAULT_DTYPE;
    }
  }

  const isComplex = isComplexDType(actualDtype);

  const Constructor = getTypedArrayConstructor(actualDtype);
  if (!Constructor) {
    throw new Error(`Cannot create array with dtype ${actualDtype}`);
  }

  const physicalSize = isComplex ? size * 2 : size;
  const typedData = new Constructor(physicalSize);
  const flatData = flattenKeepBigInt(data);

  if (isBigIntDType(actualDtype)) {
    const bigintData = typedData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      bigintData[i] = typeof val === 'bigint' ? val : BigInt(Math.round(Number(val)));
    }
  } else if (actualDtype === 'bool') {
    const boolData = typedData as Uint8Array;
    for (let i = 0; i < size; i++) {
      boolData[i] = flatData[i] ? 1 : 0;
    }
  } else if (isComplex) {
    const complexData = typedData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      let re: number, im: number;

      if (val instanceof Complex) {
        re = val.re;
        im = val.im;
      } else if (typeof val === 'object' && val !== null && 're' in val) {
        re = (val as { re: number; im?: number }).re;
        im = (val as { re: number; im?: number }).im ?? 0;
      } else {
        re = Number(val);
        im = 0;
      }

      complexData[i * 2] = re;
      complexData[i * 2 + 1] = im;
    }
  } else {
    const numData = typedData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      numData[i] = typeof val === 'bigint' ? Number(val) : Number(val);
    }
  }

  const storage = ArrayStorage.fromData(typedData, shape, actualDtype);
  return new NDArrayCore(storage);
}

/**
 * Create array with evenly spaced values within a given interval
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  let actualStart = start;
  let actualStop = stop;

  if (stop === undefined) {
    actualStart = 0;
    actualStop = start;
  }

  if (actualStop === undefined) {
    throw new Error('stop is required');
  }

  const length = Math.max(0, Math.ceil((actualStop - actualStart) / step));

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create arange array with dtype ${dtype}`);
  }

  const data = new Constructor(length);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < length; i++) {
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(actualStart + i * step));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < length; i++) {
      (data as Uint8Array)[i] = actualStart + i * step !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < length; i++) {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = actualStart + i * step;
    }
  }

  const storage = ArrayStorage.fromData(data, [length], dtype);
  return new NDArrayCore(storage);
}

/**
 * Create array with evenly spaced values over a specified interval
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([start], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create linspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const step = (stop - start) / (num - 1);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(start + i * step));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      (data as Uint8Array)[i] = start + i * step !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = start + i * step;
    }
  }

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArrayCore(storage);
}

/**
 * Create array with logarithmically spaced values
 */
export function logspace(
  start: number,
  stop: number,
  num: number = 50,
  base: number = 10.0,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([Math.pow(base, start)], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create logspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const step = (stop - start) / (num - 1);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(Math.pow(base, exponent)));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as Uint8Array)[i] = Math.pow(base, exponent) !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Math.pow(base, exponent);
    }
  }

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArrayCore(storage);
}

/**
 * Create array with geometrically spaced values
 */
export function geomspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (start === 0 || stop === 0) {
    throw new Error('Geometric sequence cannot include zero');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([start], dtype);
  }

  const signStart = Math.sign(start);
  const signStop = Math.sign(stop);

  if (signStart !== signStop) {
    throw new Error('Geometric sequence cannot contain both positive and negative values');
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create geomspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const logStart = Math.log(Math.abs(start));
  const logStop = Math.log(Math.abs(stop));
  const step = (logStop - logStart) / (num - 1);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(value));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as Uint8Array)[i] = value !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value;
    }
  }

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArrayCore(storage);
}

/**
 * Create identity matrix
 */
export function eye(
  n: number,
  m?: number,
  k: number = 0,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  const cols = m ?? n;
  const result = zeros([n, cols], dtype);
  const data = result.data;

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < cols) {
        typedData[i * cols + j] = BigInt(1);
      }
    }
  } else {
    const typedData = data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < cols) {
        typedData[i * cols + j] = 1;
      }
    }
  }

  return result;
}

/**
 * Create a square identity matrix
 */
export function identity(n: number, dtype: DType = DEFAULT_DTYPE): NDArrayCore {
  return eye(n, n, 0, dtype);
}

/**
 * Convert input to an ndarray
 */
export function asarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  if (isNDArrayLike(a)) {
    if (!dtype || a.dtype === dtype) {
      return a;
    }
    return a.astype(dtype);
  }
  return array(a, dtype);
}

/**
 * Return array of zeros with the same shape and dtype as input
 */
export function zeros_like(a: NDArrayCore, dtype?: DType): NDArrayCore {
  return zeros(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Return array of ones with the same shape and dtype as input
 */
export function ones_like(a: NDArrayCore, dtype?: DType): NDArrayCore {
  return ones(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Return empty array with the same shape and dtype as input
 */
export function empty_like(a: NDArrayCore, dtype?: DType): NDArrayCore {
  return empty(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Return array filled with value, same shape and dtype as input
 */
export function full_like(
  a: NDArrayCore,
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArrayCore {
  return full(Array.from(a.shape), fill_value, dtype ?? (a.dtype as DType));
}

/**
 * Deep copy of array
 */
export function copy(a: NDArrayCore): NDArrayCore {
  return a.copy();
}

// ============================================================
// Array Conversion
// ============================================================

export function asanyarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  return asarray(a as NDArrayCore, dtype);
}

export function ascontiguousarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  const arr = asarray(a as NDArrayCore, dtype);
  // In our implementation, arrays are always contiguous
  return arr.copy();
}

export function asfortranarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  const arr = asarray(a as NDArrayCore, dtype);
  // Note: We don't actually support Fortran order, return C-order copy
  return arr.copy();
}

export function asarray_chkfinite(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  const arr = asarray(a as NDArrayCore, dtype);
  const data = arr.data;

  for (let i = 0; i < data.length; i++) {
    const val = data[i] as number;
    if (!Number.isFinite(val)) {
      throw new Error('array must not contain infs or NaNs');
    }
  }

  return arr;
}

export function require(
  a: NDArrayCore,
  dtype?: DType,
  _requirements?: string | string[]
): NDArrayCore {
  let result = a;
  if (dtype && dtype !== a.dtype) {
    result = result.astype(dtype);
  }
  // Requirements like 'C', 'F', 'A', 'W', 'O', 'E' are mostly no-ops in our implementation
  return result;
}

// ============================================================
// Diagonal and Triangular
// ============================================================

// Helper: flatten an NDArrayCore to 1D
function flattenCore(a: NDArrayCore): NDArrayCore {
  const data = a.data;
  const storage = ArrayStorage.fromData(
    data.slice() as typeof data,
    [data.length],
    a.dtype as DType
  );
  return fromStorage(storage);
}

export function diag(v: NDArrayCore, k: number = 0): NDArrayCore {
  const shape = v.shape;
  const data = v.data;
  const dtype = v.dtype as DType;

  if (shape.length === 1) {
    // Create a 2D array with v as diagonal
    const dim0 = shape[0]!;
    const n = dim0 + Math.abs(k);
    const result = zeros([n, n], dtype);
    const resultData = result.data;

    for (let i = 0; i < dim0; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      if (row >= 0 && row < n && col >= 0 && col < n) {
        (resultData as Float64Array)[row * n + col] = data[i] as number;
      }
    }
    return result;
  } else if (shape.length === 2) {
    // Extract diagonal from 2D array
    const rows = shape[0]!;
    const cols = shape[1]!;
    const diagLen = Math.min(
      k >= 0 ? Math.min(rows, cols - k) : Math.min(rows + k, cols),
      Math.max(0, k >= 0 ? cols - k : rows + k)
    );

    if (diagLen <= 0) {
      return array([], dtype);
    }

    const resultArr: number[] = [];
    for (let i = 0; i < diagLen; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      if (row >= 0 && row < rows && col >= 0 && col < cols) {
        resultArr.push(data[row * cols + col] as number);
      }
    }
    return array(resultArr, dtype);
  }

  throw new Error('Input must be 1-D or 2-D');
}

export function diagflat(v: NDArrayCore, k: number = 0): NDArrayCore {
  // Flatten v first, then create diagonal matrix
  const flat = flattenCore(v);
  return diag(flat, k);
}

export function tri(
  N: number,
  M?: number,
  k: number = 0,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  const cols = M ?? N;
  const result = zeros([N, cols], dtype);
  const data = result.data;

  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= Math.min(i + k, cols - 1); j++) {
      if (j >= 0) {
        (data as Float64Array)[i * cols + j] = 1;
      }
    }
  }

  return result;
}

export function tril(m: NDArrayCore, k: number = 0): NDArrayCore {
  const shape = m.shape;
  if (shape.length < 2) {
    throw new Error('Input must be at least 2-D');
  }

  const result = m.copy();
  const data = result.data;
  const rows = shape[shape.length - 2]!;
  const cols = shape[shape.length - 1]!;
  const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
  const matrixSize = rows * cols;

  for (let b = 0; b < batchSize; b++) {
    const offset = b * matrixSize;
    for (let i = 0; i < rows; i++) {
      for (let j = i + k + 1; j < cols; j++) {
        (data as Float64Array)[offset + i * cols + j] = 0;
      }
    }
  }

  return result;
}

export function triu(m: NDArrayCore, k: number = 0): NDArrayCore {
  const shape = m.shape;
  if (shape.length < 2) {
    throw new Error('Input must be at least 2-D');
  }

  const result = m.copy();
  const data = result.data;
  const rows = shape[shape.length - 2]!;
  const cols = shape[shape.length - 1]!;
  const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
  const matrixSize = rows * cols;

  for (let b = 0; b < batchSize; b++) {
    const offset = b * matrixSize;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < Math.min(i + k, cols); j++) {
        (data as Float64Array)[offset + i * cols + j] = 0;
      }
    }
  }

  return result;
}

export function vander(x: NDArrayCore, N?: number, increasing: boolean = false): NDArrayCore {
  const n = x.size;
  const cols = N ?? n;
  const data = x.data;
  const result = zeros([n, cols], x.dtype as DType);
  const resultData = result.data;

  for (let i = 0; i < n; i++) {
    const val = data[i] as number;
    for (let j = 0; j < cols; j++) {
      const exp = increasing ? j : cols - 1 - j;
      (resultData as Float64Array)[i * cols + j] = Math.pow(val, exp);
    }
  }

  return result;
}

// ============================================================
// From Data Sources
// ============================================================

export function frombuffer(
  buffer: ArrayBuffer | TypedArray,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  offset: number = 0
): NDArrayCore {
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }

  let data: TypedArray;
  if (buffer instanceof ArrayBuffer) {
    // offset is in bytes for ArrayBuffer
    const byteOffset = offset;
    const length =
      count < 0 ? (buffer.byteLength - byteOffset) / Constructor.BYTES_PER_ELEMENT : count;
    data = new Constructor(buffer, byteOffset, length);
  } else {
    const start = offset;
    const end = count < 0 ? buffer.length : offset + count;
    // Extract values from source buffer and create new typed array
    const sliced = Array.from(buffer.slice(start, end) as ArrayLike<number | bigint>);
    data = new Constructor(sliced.length);
    for (let i = 0; i < sliced.length; i++) {
      (data as unknown as number[])[i] = sliced[i] as number;
    }
  }

  const storage = ArrayStorage.fromData(data, [data.length], dtype);
  return fromStorage(storage);
}

export function fromfunction(
  func: (...indices: number[]) => number,
  shape: number[],
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  const size = shape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }

  const data = new Constructor(size);
  const strides: number[] = [];
  let strideVal = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(strideVal);
    strideVal *= shape[i]!;
  }

  for (let flatIdx = 0; flatIdx < size; flatIdx++) {
    const indices: number[] = [];
    let remaining = flatIdx;
    for (let i = 0; i < shape.length; i++) {
      indices.push(Math.floor(remaining / strides[i]!));
      remaining = remaining % strides[i]!;
    }
    (data as Float64Array)[flatIdx] = func(...indices);
  }

  const storage = ArrayStorage.fromData(data, shape, dtype);
  return fromStorage(storage);
}

export function fromiter(
  iter: Iterable<number>,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1
): NDArrayCore {
  const values: number[] = [];
  let i = 0;
  for (const val of iter) {
    if (count >= 0 && i >= count) break;
    values.push(val);
    i++;
  }
  return array(values, dtype);
}

export function fromstring(
  string: string,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  sep?: string
): NDArrayCore {
  // Default to whitespace splitting (NumPy uses sep='' for binary, but we default to whitespace for convenience)
  const separator = sep ?? /\s+/;

  const parts = string.split(separator).filter((s) => s.trim() !== '');
  const values = parts.map((s) => parseFloat(s.trim()));
  const actualValues = count >= 0 ? values.slice(0, count) : values;

  return array(actualValues, dtype);
}

export function fromfile(
  _file: string,
  _dtype: DType = DEFAULT_DTYPE,
  _count: number = -1,
  _sep: string = ''
): NDArrayCore {
  throw new Error('fromfile requires Node.js file system access');
}

export function meshgrid(...arrays: NDArrayCore[]): NDArrayCore[] {
  if (arrays.length === 0) return [];
  if (arrays.length === 1) return [arrays[0]!.copy()];

  const shapes = arrays.map((a) => a.size);
  const outputShape = [...shapes];

  const results: NDArrayCore[] = [];

  for (let dim = 0; dim < arrays.length; dim++) {
    const arr = arrays[dim]!;
    const data = arr.data;
    const result = zeros(outputShape, arr.dtype as DType);
    const resultData = result.data;

    // Calculate strides for output
    const strides: number[] = [];
    let strideVal = 1;
    for (let i = outputShape.length - 1; i >= 0; i--) {
      strides.unshift(strideVal);
      strideVal *= outputShape[i]!;
    }

    const totalSize = outputShape.reduce((a, b) => a * b, 1);
    for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
      // Get the index along this dimension
      const idx = Math.floor(flatIdx / strides[dim]!) % shapes[dim]!;
      (resultData as Float64Array)[flatIdx] = data[idx] as number;
    }

    results.push(result);
  }

  return results;
}
