/**
 * Rounding operations
 *
 * Pure functions for element-wise rounding operations:
 * around, ceil, fix, floor, rint, round, trunc
 *
 * Note: Rounding operations are not defined for complex numbers.
 * All functions throw TypeError for complex dtypes.
 */

import type { Complex } from '../complex';
import { isComplexDType, isIntegerDType, mathResultDtype, throwIfComplex } from '../dtype';
import { ArrayStorage } from '../storage';
import { wasmAround, wasmCeil, wasmFloor, wasmRint, wasmTrunc } from '../wasm/rounding';

/**
 * Apply a rounding function component-wise to a complex array.
 * NumPy applies rounding to real and imaginary parts independently for rint/around.
 */
function complexComponentwise(a: ArrayStorage, fn: (x: number) => number): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;
  const result = ArrayStorage.empty(shape, dtype);
  const dstData = result.data as Float64Array | Float32Array;

  if (a.isCContiguous) {
    const srcData = a.data as Float64Array | Float32Array;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      dstData[i * 2] = fn(srcData[(off + i) * 2]!);
      dstData[i * 2 + 1] = fn(srcData[(off + i) * 2 + 1]!);
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = a.iget(i) as Complex;
      dstData[i * 2] = fn(val.re);
      dstData[i * 2 + 1] = fn(val.im);
    }
  }

  return result;
}

/**
 * Round half to even (banker's rounding) — matches NumPy's `rint`.
 *
 * The tie test has to be exact. An earlier version treated anything within 1e-10
 * of .5 as a tie, which disagreed with NumPy on near-ties: `np.rint(2.5000000000001)`
 * is 3, not 2. It also disagreed with the WASM kernel that now handles arrays of
 * 32 elements or more, so the same value could round differently depending on
 * array size.
 *
 * `Math.floor` and the subtraction are both exact, so `frac === 0.5` identifies
 * a genuine tie and nothing else. For |x| >= 2^52 the fraction is 0 and x falls
 * out of the first branch unchanged.
 */
function roundHalfToEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const lo = Math.floor(x);
  const frac = x - lo;
  let r: number;
  if (frac < 0.5) r = lo;
  else if (frac > 0.5) r = lo + 1;
  // Exact tie: pick whichever neighbour is even. `lo % 2` is -0 for negative
  // even values, and -0 === 0, so this holds for both signs.
  else r = lo % 2 === 0 ? lo : lo + 1;
  // NumPy keeps the sign of zero — rint(-0.5) is -0.0, not +0.0 — and so does
  // the WASM kernel, which re-applies the input's sign bit. Without this the two
  // paths disagreed on any negative input that rounds to zero.
  return r === 0 && (x < 0 || Object.is(x, -0)) ? -0 : r;
}

/**
 * Round an array to the given number of decimals
 */
export function around(a: ArrayStorage, decimals: number = 0): ArrayStorage {
  if (isComplexDType(a.dtype)) {
    const multiplier = 10 ** decimals;
    return complexComponentwise(a, (x) => roundHalfToEven(x * multiplier) / multiplier);
  }
  if (isIntegerDType(a.dtype) && decimals >= 0) return a.copy();
  // decimals === 0 is plain rint; anything else needs the scaled kernel.
  const wasmAroundResult = decimals === 0 ? wasmRint(a) : wasmAround(a, 10 ** decimals);
  if (wasmAroundResult) return wasmAroundResult;
  if (a.dtype === 'bool') {
    const dt = mathResultDtype('bool');
    const r = ArrayStorage.empty(Array.from(a.shape), dt);
    const src = a.data as Uint8Array;
    const off = a.offset;
    for (let i = 0; i < a.size; i++) r.data[i] = src[off + i]!;
    return r;
  }
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  const multiplier = 10 ** decimals;

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
  if (isIntegerDType(a.dtype)) return a.copy();
  const wasmResult = wasmCeil(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
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
  if (isIntegerDType(a.dtype)) return a.copy();
  // fix is trunc — same kernel.
  const wasmResult = wasmTrunc(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
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
  if (isIntegerDType(a.dtype)) return a.copy();
  const wasmResult = wasmFloor(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
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
  if (isComplexDType(a.dtype)) return complexComponentwise(a, roundHalfToEven);
  // NumPy: rint promotes ints/bool via mathResultDtype (values stay the same, just cast)
  if (isIntegerDType(a.dtype) || a.dtype === 'bool') {
    const dt = mathResultDtype(a.dtype);
    const r = ArrayStorage.empty(Array.from(a.shape), dt);
    const src = a.data;
    const off = a.offset;
    for (let i = 0; i < a.size; i++) r.data[i] = Number(src[off + i]!);
    return r;
  }
  const wasmResult = wasmRint(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
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
  if (isIntegerDType(a.dtype)) return a.copy();
  const wasmResult = wasmTrunc(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
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
