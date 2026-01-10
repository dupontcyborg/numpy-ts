/**
 * Array Creation Module
 *
 * This module provides array creation functions that do NOT depend on any ops modules.
 * Importing from this module will not pull in arithmetic, linalg, fft, random, etc.
 *
 * For tree-shakeable imports:
 *   import { zeros, ones, array } from 'numpy-ts/creation';
 */

import { ArrayStorage } from '../core/storage';
import { NDArrayCore, type DType, type TypedArray } from '../core/ndarray-core';
import { Complex, isComplexLike } from '../core/complex';
import { getTypedArrayConstructor, isBigIntDType, isComplexDType, DEFAULT_DTYPE } from '../core/dtype';

// Re-export NDArrayCore as the array type for this module
export { NDArrayCore as NDArray } from '../core/ndarray-core';
export type { DType, TypedArray } from '../core/ndarray-core';

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

// Helper functions
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
export function eye(n: number, m?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArrayCore {
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
  if (a instanceof NDArrayCore) {
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
