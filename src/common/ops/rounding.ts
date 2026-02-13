/**
 * Rounding operations
 *
 * Pure functions for element-wise rounding operations:
 * around, ceil, fix, floor, rint, round, trunc
 *
 * Note: Rounding operations are not defined for complex numbers.
 * All functions throw TypeError for complex dtypes.
 */

import { ArrayStorage } from '../storage';
import { throwIfComplex } from '../dtype';

/**
 * Round half to even (banker's rounding) - matches NumPy behavior
 */
function roundHalfToEven(x: number): number {
  if (!isFinite(x)) return x;
  const floor = Math.floor(x);
  const decimal = x - floor;
  // If exactly 0.5, round to nearest even
  if (Math.abs(decimal - 0.5) < 1e-10) {
    return floor % 2 === 0 ? floor : floor + 1;
  }
  return Math.round(x);
}

/**
 * Round an array to the given number of decimals
 */
export function around(a: ArrayStorage, decimals: number = 0): ArrayStorage {
  throwIfComplex(a.dtype, 'around', 'Rounding is not defined for complex numbers.');
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  const multiplier = Math.pow(10, decimals);

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      const val = Number(data[off + i]!);
      resultData[i] = roundHalfToEven(val * multiplier) / multiplier;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = Number(a.iget(i));
      resultData[i] = roundHalfToEven(val * multiplier) / multiplier;
    }
  }

  return result;
}

/**
 * Return the ceiling of the input, element-wise
 */
export function ceil(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'ceil', 'Rounding is not defined for complex numbers.');
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.ceil(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.ceil(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Round to nearest integer towards zero
 */
export function fix(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'fix', 'Rounding is not defined for complex numbers.');
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Return the floor of the input, element-wise
 */
export function floor(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'floor', 'Rounding is not defined for complex numbers.');
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.floor(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.floor(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Round elements of the array to the nearest integer (banker's rounding)
 */
export function rint(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'rint', 'Rounding is not defined for complex numbers.');
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = roundHalfToEven(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = roundHalfToEven(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Alias for around
 */
export function round(a: ArrayStorage, decimals: number = 0): ArrayStorage {
  return around(a, decimals);
}

/**
 * Return the truncated value of the input, element-wise
 */
export function trunc(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'trunc', 'Rounding is not defined for complex numbers.');
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(a.iget(i)));
    }
  }

  return result;
}
