// AUTO-GENERATED - DO NOT EDIT
// Run `npm run generate` to regenerate this file from core/ modules

/**
 * Full Module - NDArray with method chaining
 *
 * This module wraps core/ functions to return NDArray (with methods)
 * instead of NDArrayCore (minimal). Use this for method chaining.
 *
 * @example
 * ```typescript
 * import { array } from 'numpy-ts/full';
 *
 * const result = array([1, 2, 3]).add(10).reshape([3, 1]);
 * ```
 */

import * as core from '../core';
import { NDArray } from './ndarray';
import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import { Complex } from '../common/complex';
import type { DType, TypedArray } from '../core/types';
import {
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  promoteDTypes,
  isComplexDType,
} from '../common/dtype';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
// If already an NDArray, return as-is to preserve identity
// Also preserves base reference for views
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

// Re-export types
export type { DType, TypedArray } from '../core/types';
export { NDArray, meshgrid } from './ndarray';
export { NDArrayCore } from '../common/ndarray-core';
export { Complex } from '../common/complex';

// ============================================================
// Wrapped Functions (return NDArray)
// ============================================================

/** Broadcast array to a new shape - returns a view */
export function broadcast_to(a: NDArrayCore, shape: number[]): NDArray {
  return up(core.broadcast_to(a, shape));
}

export function take(a: NDArrayCore, indices: number[], axis?: number): NDArray {
  return up(core.take(a, indices, axis));
}

export function take_along_axis(arr: NDArrayCore, indices: NDArrayCore, axis: number): NDArray {
  return up(core.take_along_axis(arr, indices, axis));
}

export function choose(a: NDArrayCore, choices: NDArrayCore[]): NDArray {
  return up(core.choose(a, choices));
}

export function compress(condition: NDArrayCore, a: NDArrayCore, axis?: number): NDArray {
  return up(core.compress(condition, a, axis));
}

/**
 * Integer array indexing (fancy indexing)
 *
 * Select elements using an array of indices.
 */
export function iindex(
  a: NDArrayCore,
  indices: NDArrayCore | number[] | number[][],
  axis: number = 0
): NDArray {
  return up(core.iindex(a, indices, axis));
}

/**
 * Boolean array indexing (fancy indexing with mask)
 *
 * Select elements where a boolean mask is true.
 */
export function bindex(a: NDArrayCore, mask: NDArrayCore, axis?: number): NDArray {
  return up(core.bindex(a, mask, axis));
}

export function select(
  condlist: NDArrayCore[],
  choicelist: NDArrayCore[],
  defaultVal: number = 0
): NDArray {
  return up(core.select(condlist, choicelist, defaultVal));
}

export function indices(
  dimensions: number[],
  dtype: 'int32' | 'int64' | 'float64' = 'int32'
): NDArray {
  return up(core.indices(dimensions, dtype));
}

export function ravel_multi_index(
  multi_index: NDArrayCore[],
  dims: number[],
  mode: 'raise' | 'wrap' | 'clip' = 'raise'
): NDArray {
  return up(core.ravel_multi_index(multi_index, dims, mode));
}

export function unravel_index(indices: NDArrayCore | number, shape: number[]): NDArray[] {
  return core.unravel_index(indices, shape).map(up);
}

export function diag_indices(n: number, ndim: number = 2): NDArray[] {
  return core.diag_indices(n, ndim).map(up);
}

export function diag_indices_from(a: NDArrayCore): NDArray[] {
  return core.diag_indices_from(a).map(up);
}

export function tril_indices(n: number, k: number = 0, m?: number): NDArray[] {
  return core.tril_indices(n, k, m).map(up);
}

export function tril_indices_from(a: NDArrayCore, k: number = 0): NDArray[] {
  return core.tril_indices_from(a, k).map(up);
}

export function triu_indices(n: number, k: number = 0, m?: number): NDArray[] {
  return core.triu_indices(n, k, m).map(up);
}

export function triu_indices_from(a: NDArrayCore, k: number = 0): NDArray[] {
  return core.triu_indices_from(a, k).map(up);
}

export function mask_indices(
  n: number,
  mask_func: (n: number, k: number) => NDArrayCore,
  k: number = 0
): NDArray[] {
  return core.mask_indices(n, mask_func, k).map(up);
}

export function apply_along_axis(
  func1d: (arr: NDArrayCore) => NDArrayCore | number,
  axis: number,
  arr: NDArrayCore
): NDArray {
  const wrappedFunc1d = (arr: NDArrayCore): NDArrayCore | number => {
    return func1d(up(arr));
  };
  return up(core.apply_along_axis(wrappedFunc1d, axis, arr));
}

export function apply_over_axes(
  func: (arr: NDArrayCore, axis: number) => NDArrayCore,
  a: NDArrayCore,
  axes: number[]
): NDArray {
  const wrappedFunc = (arr: NDArrayCore, axis: number): NDArrayCore => {
    return func(up(arr), axis);
  };
  return up(core.apply_over_axes(wrappedFunc, a, axes));
}

/** Add arguments element-wise */
export function add(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.add(x1, x2));
}

/** Subtract arguments element-wise */
export function subtract(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.subtract(x1, x2));
}

/** Multiply arguments element-wise */
export function multiply(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.multiply(x1, x2));
}

/** Square root of array elements */
export function sqrt(x: NDArrayCore): NDArray {
  return up(core.sqrt(x));
}

/** Element-wise power */
export function power(x: NDArrayCore, exponent: NDArrayCore | number): NDArray {
  return up(core.power(x, exponent));
}

/** Exponential (e^x) */
export function exp(x: NDArrayCore): NDArray {
  return up(core.exp(x));
}

/** 2^x */
export function exp2(x: NDArrayCore): NDArray {
  return up(core.exp2(x));
}

/** exp(x) - 1 */
export function expm1(x: NDArrayCore): NDArray {
  return up(core.expm1(x));
}

/** Natural logarithm */
export function log(x: NDArrayCore): NDArray {
  return up(core.log(x));
}

/** Base-2 logarithm */
export function log2(x: NDArrayCore): NDArray {
  return up(core.log2(x));
}

/** Base-10 logarithm */
export function log10(x: NDArrayCore): NDArray {
  return up(core.log10(x));
}

/** log(1 + x) */
export function log1p(x: NDArrayCore): NDArray {
  return up(core.log1p(x));
}

/** log(exp(x1) + exp(x2)) */
export function logaddexp(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logaddexp(x1, x2));
}

/** log2(2^x1 + 2^x2) */
export function logaddexp2(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logaddexp2(x1, x2));
}

/** Absolute value */
export function absolute(x: NDArrayCore): NDArray {
  return up(core.absolute(x));
}

/** Numerical negative */
export function negative(x: NDArrayCore): NDArray {
  return up(core.negative(x));
}

/** Element-wise sign */
export function sign(x: NDArrayCore): NDArray {
  return up(core.sign(x));
}

/** Element-wise modulo */
export function mod(x: NDArrayCore, divisor: NDArrayCore | number): NDArray {
  return up(core.mod(x, divisor));
}

/** Element-wise division */
export function divide(x: NDArrayCore, divisor: NDArrayCore | number): NDArray {
  return up(core.divide(x, divisor));
}

/** Element-wise floor division */
export function floor_divide(x: NDArrayCore, divisor: NDArrayCore | number): NDArray {
  return up(core.floor_divide(x, divisor));
}

/** Numerical positive (returns copy) */
export function positive(x: NDArrayCore): NDArray {
  return up(core.positive(x));
}

/** Reciprocal (1/x) */
export function reciprocal(x: NDArrayCore): NDArray {
  return up(core.reciprocal(x));
}

/** Cube root */
export function cbrt(x: NDArrayCore): NDArray {
  return up(core.cbrt(x));
}

/** Absolute value (float) */
export function fabs(x: NDArrayCore): NDArray {
  return up(core.fabs(x));
}

/** Element-wise divmod (quotient and remainder) */
export function divmod(x: NDArrayCore, y: NDArrayCore | number): [NDArray, NDArray] {
  const r = core.divmod(x, y);
  return [up(r[0]), up(r[1])] as [NDArray, NDArray];
}

/** Element-wise square */
export function square(x: NDArrayCore): NDArray {
  return up(core.square(x));
}

/** Element-wise remainder */
export function remainder(x: NDArrayCore, y: NDArrayCore | number): NDArray {
  return up(core.remainder(x, y));
}

/** Heaviside step function */
export function heaviside(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.heaviside(x1, x2));
}

/** Float power (always returns float) */
export function float_power(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.float_power(x1, x2));
}

/** C-style fmod */
export function fmod(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.fmod(x1, x2));
}

/** Decompose into mantissa and exponent */
export function frexp(x: NDArrayCore): [NDArray, NDArray] {
  const r = core.frexp(x);
  return [up(r[0]), up(r[1])] as [NDArray, NDArray];
}

/** Greatest common divisor */
export function gcd(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.gcd(x1, x2));
}

/** Least common multiple */
export function lcm(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.lcm(x1, x2));
}

/** Compose from mantissa and exponent */
export function ldexp(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.ldexp(x1, x2));
}

/** Decompose into fractional and integer parts */
export function modf(x: NDArrayCore): [NDArray, NDArray] {
  const r = core.modf(x);
  return [up(r[0]), up(r[1])] as [NDArray, NDArray];
}

/** Clip values to a range */
export function clip(
  a: NDArrayCore,
  a_min: NDArrayCore | number | null,
  a_max: NDArrayCore | number | null
): NDArray {
  return up(core.clip(a, a_min, a_max));
}

/** Element-wise maximum */
export function maximum(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.maximum(x1, x2));
}

/** Element-wise minimum */
export function minimum(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.minimum(x1, x2));
}

/** Element-wise maximum (ignores NaN) */
export function fmax(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.fmax(x1, x2));
}

/** Element-wise minimum (ignores NaN) */
export function fmin(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.fmin(x1, x2));
}

/** Replace NaN/Inf with numbers */
export function nan_to_num(
  x: NDArrayCore,
  nan: number = 0.0,
  posinf?: number,
  neginf?: number
): NDArray {
  return up(core.nan_to_num(x, nan, posinf, neginf));
}

/** 1D linear interpolation */
export function interp(
  x: NDArrayCore,
  xp: NDArrayCore,
  fp: NDArrayCore,
  left?: number,
  right?: number
): NDArray {
  return up(core.interp(x, xp, fp, left, right));
}

/** Unwrap phase angles */
export function unwrap(
  p: NDArrayCore,
  discont: number = Math.PI,
  axis: number = -1,
  period: number = 2 * Math.PI
): NDArray {
  return up(core.unwrap(p, discont, axis, period));
}

/** Sinc function */
export function sinc(x: NDArrayCore): NDArray {
  return up(core.sinc(x));
}

/** Modified Bessel function of the first kind, order 0 */
export function i0(x: NDArrayCore): NDArray {
  return up(core.i0(x));
}

/** Bitwise AND */
export function bitwise_and(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_and(x1, x2));
}

/** Bitwise OR */
export function bitwise_or(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_or(x1, x2));
}

/** Bitwise XOR */
export function bitwise_xor(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_xor(x1, x2));
}

/** Bitwise NOT */
export function bitwise_not(x: NDArrayCore): NDArray {
  return up(core.bitwise_not(x));
}

/** Bitwise inversion (alias for bitwise_not) */
export function invert(x: NDArrayCore): NDArray {
  return up(core.invert(x));
}

/** Left shift */
export function left_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.left_shift(x1, x2));
}

/** Right shift */
export function right_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.right_shift(x1, x2));
}

/** Pack bits into bytes */
export function packbits(a: NDArrayCore, axis?: number, bitorder?: 'big' | 'little'): NDArray {
  return up(core.packbits(a, axis, bitorder));
}

/** Unpack bytes into bits */
export function unpackbits(
  a: NDArrayCore,
  axis?: number,
  count?: number,
  bitorder?: 'big' | 'little'
): NDArray {
  return up(core.unpackbits(a, axis, count, bitorder));
}

/** Count number of 1 bits */
export function bitwise_count(x: NDArrayCore): NDArray {
  return up(core.bitwise_count(x));
}

/** Alias for bitwise_not */
export function bitwise_invert(x: NDArrayCore): NDArray {
  return up(core.bitwise_invert(x));
}

/** Alias for left_shift */
export function bitwise_left_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_left_shift(x1, x2));
}

/** Alias for right_shift */
export function bitwise_right_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_right_shift(x1, x2));
}

/** Extract real part of array */
export function real(x: NDArrayCore): NDArray {
  return up(core.real(x));
}

/** Extract imaginary part of array */
export function imag(x: NDArrayCore): NDArray {
  return up(core.imag(x));
}

/** Complex conjugate */
export function conj(x: NDArrayCore): NDArray {
  return up(core.conj(x));
}

/** Phase angle */
export function angle(x: NDArrayCore, deg?: boolean): NDArray {
  return up(core.angle(x, deg));
}

/**
 * Create array of zeros
 */
export function zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  return up(core.zeros(shape, dtype));
}

/**
 * Create array of ones
 */
export function ones(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  return up(core.ones(shape, dtype));
}

/**
 * Create an uninitialized array (zeros in JS)
 */
export function empty(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  return up(core.empty(shape, dtype));
}

/**
 * Create array filled with a constant value
 */
export function full(
  shape: number[],
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  return up(core.full(shape, fill_value, dtype));
}

/**
 * Create array from nested JavaScript arrays
 */
export function array(data: unknown, dtype?: DType): NDArray {
  return up(core.array(data, dtype));
}

/**
 * Create array with evenly spaced values within a given interval
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  return up(core.arange(start, stop, step, dtype));
}

/**
 * Create array with evenly spaced values over a specified interval
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  return up(core.linspace(start, stop, num, dtype));
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
): NDArray {
  return up(core.logspace(start, stop, num, base, dtype));
}

/**
 * Create array with geometrically spaced values
 */
export function geomspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  return up(core.geomspace(start, stop, num, dtype));
}

/**
 * Create identity matrix
 */
export function eye(n: number, m?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArray {
  return up(core.eye(n, m, k, dtype));
}

/**
 * Create a square identity matrix
 */
export function identity(n: number, dtype: DType = DEFAULT_DTYPE): NDArray {
  return up(core.identity(n, dtype));
}

/**
 * Convert input to an ndarray
 */
export function asarray(a: NDArrayCore | unknown, dtype?: DType): NDArray {
  return up(core.asarray(a, dtype));
}

/**
 * Return array of zeros with the same shape and dtype as input
 */
export function zeros_like(a: NDArrayCore, dtype?: DType): NDArray {
  return up(core.zeros_like(a, dtype));
}

/**
 * Return array of ones with the same shape and dtype as input
 */
export function ones_like(a: NDArrayCore, dtype?: DType): NDArray {
  return up(core.ones_like(a, dtype));
}

/**
 * Return empty array with the same shape and dtype as input
 */
export function empty_like(a: NDArrayCore, dtype?: DType): NDArray {
  return up(core.empty_like(a, dtype));
}

/**
 * Return array filled with value, same shape and dtype as input
 */
export function full_like(
  a: NDArrayCore,
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  return up(core.full_like(a, fill_value, dtype));
}

/**
 * Deep copy of array
 */
export function copy(a: NDArrayCore): NDArray {
  return up(core.copy(a));
}

export function asanyarray(a: NDArrayCore | unknown, dtype?: DType): NDArray {
  return up(core.asanyarray(a, dtype));
}

export function ascontiguousarray(a: NDArrayCore | unknown, dtype?: DType): NDArray {
  return up(core.ascontiguousarray(a, dtype));
}

export function asfortranarray(a: NDArrayCore | unknown, dtype?: DType): NDArray {
  return up(core.asfortranarray(a, dtype));
}

export function asarray_chkfinite(a: NDArrayCore | unknown, dtype?: DType): NDArray {
  return up(core.asarray_chkfinite(a, dtype));
}

export function require(a: NDArrayCore, dtype?: DType, _requirements?: string | string[]): NDArray {
  return up(core.require(a, dtype, _requirements));
}

export function diag(v: NDArrayCore, k: number = 0): NDArray {
  return up(core.diag(v, k));
}

export function diagflat(v: NDArrayCore, k: number = 0): NDArray {
  return up(core.diagflat(v, k));
}

export function tri(N: number, M?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArray {
  return up(core.tri(N, M, k, dtype));
}

export function tril(m: NDArrayCore, k: number = 0): NDArray {
  return up(core.tril(m, k));
}

export function triu(m: NDArrayCore, k: number = 0): NDArray {
  return up(core.triu(m, k));
}

export function vander(x: NDArrayCore, N?: number, increasing: boolean = false): NDArray {
  return up(core.vander(x, N, increasing));
}

export function frombuffer(
  buffer: ArrayBuffer | TypedArray,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  offset: number = 0
): NDArray {
  return up(core.frombuffer(buffer, dtype, count, offset));
}

export function fromfunction(
  func: (...indices: number[]) => number,
  shape: number[],
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  return up(core.fromfunction(func, shape, dtype));
}

export function fromiter(
  iter: Iterable<number>,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1
): NDArray {
  return up(core.fromiter(iter, dtype, count));
}

export function fromstring(
  string: string,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  sep?: string
): NDArray {
  return up(core.fromstring(string, dtype, count, sep));
}

export function fromfile(
  _file: string,
  _dtype: DType = DEFAULT_DTYPE,
  _count: number = -1,
  _sep: string = ''
): NDArray {
  return up(core.fromfile(_file, _dtype, _count, _sep));
}

/** Calculate n-th discrete difference */
export function diff(a: NDArrayCore, n?: number, axis?: number): NDArray {
  return up(core.diff(a, n, axis));
}

/** Difference between consecutive elements in 1D array */
export function ediff1d(
  a: NDArrayCore,
  to_end?: number[] | null,
  to_begin?: number[] | null
): NDArray {
  return up(core.ediff1d(a, to_end, to_begin));
}

/** Dot product of two arrays */
export function dot(a: NDArrayCore, b: NDArrayCore): NDArray | number | bigint | Complex {
  const r = core.dot(a, b);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Return diagonal or construct diagonal array */
export function diagonal(
  a: NDArrayCore,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1
): NDArray {
  return up(core.diagonal(a, offset, axis1, axis2));
}

/** Kronecker product */
export function kron(a: NDArrayCore, b: NDArrayCore): NDArray {
  return up(core.kron(a, b));
}

/** Transpose array - returns a view */
export function transpose(a: NDArrayCore, axes?: number[]): NDArray {
  return up(core.transpose(a, axes));
}

/** Inner product of two arrays */
export function inner(a: NDArrayCore, b: NDArrayCore): NDArray | number | bigint | Complex {
  const r = core.inner(a, b);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Outer product of two arrays */
export function outer(a: NDArrayCore, b: NDArrayCore): NDArray {
  return up(core.outer(a, b));
}

/** Tensor dot product */
export function tensordot(
  a: NDArrayCore,
  b: NDArrayCore,
  axes: number | [number[], number[]] = 2
): NDArray | number | bigint | Complex {
  const r = core.tensordot(a, b, axes);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Vector dot product along specified axis */
export function vecdot(
  x1: NDArrayCore,
  x2: NDArrayCore,
  axis: number = -1
): NDArray | number | bigint | Complex {
  const r = core.vecdot(x1, x2, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Matrix transpose (swap last two axes) */
/** Matrix transpose - returns a view */
export function matrix_transpose(a: NDArrayCore): NDArray {
  return up(core.matrix_transpose(a));
}

/** Permute array dimensions - returns a view */
export function permute_dims(a: NDArrayCore, axes?: number[]): NDArray {
  return up(core.permute_dims(a, axes));
}

/** Matrix-vector product */
export function matvec(x1: NDArrayCore, x2: NDArrayCore): NDArray {
  return up(core.matvec(x1, x2));
}

/** Vector-matrix product */
export function vecmat(x1: NDArrayCore, x2: NDArrayCore): NDArray {
  return up(core.vecmat(x1, x2));
}

/** Cross product */
export function cross(
  a: NDArrayCore,
  b: NDArrayCore,
  axisa: number = -1,
  axisb: number = -1,
  axisc: number = -1,
  axis?: number
): NDArray {
  const r = core.cross(a, b, axisa, axisb, axisc, axis);
  const dtype = promoteDTypes(a.dtype as DType, b.dtype as DType);
  if (r instanceof Complex) {
    const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
    const Ctor = getTypedArrayConstructor(baseDtype)!;
    const data = new Ctor(2);
    data[0] = r.re;
    data[1] = r.im;
    return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
  }
  if (typeof r === 'number') {
    if (isComplexDType(dtype)) {
      const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
      const Ctor = getTypedArrayConstructor(baseDtype)!;
      const data = new Ctor(2);
      data[0] = r;
      data[1] = 0;
      return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
    }
    const Ctor = getTypedArrayConstructor(dtype)!;
    const data = new Ctor(1);
    data[0] = r as never;
    return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
  }
  return up(r);
}

/** Matrix multiplication */
export function matmul(a: NDArrayCore, b: NDArrayCore): NDArray {
  return up(core.matmul(a, b));
}

/** Element-wise logical AND */
export function logical_and(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logical_and(x1, x2));
}

/** Element-wise logical OR */
export function logical_or(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logical_or(x1, x2));
}

/** Element-wise logical NOT */
export function logical_not(x: NDArrayCore): NDArray {
  return up(core.logical_not(x));
}

/** Element-wise logical XOR */
export function logical_xor(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logical_xor(x1, x2));
}

/** Test for finite values */
export function isfinite(x: NDArrayCore): NDArray {
  return up(core.isfinite(x));
}

/** Test for infinity */
export function isinf(x: NDArrayCore): NDArray {
  return up(core.isinf(x));
}

/** Test for NaN */
export function isnan(x: NDArrayCore): NDArray {
  return up(core.isnan(x));
}

/** Test for NaT (not a time) - returns all False for numeric arrays */
export function isnat(x: NDArrayCore): NDArray {
  return up(core.isnat(x));
}

/** Test for negative infinity */
export function isneginf(x: NDArrayCore): NDArray {
  return up(core.isneginf(x));
}

/** Test for positive infinity */
export function isposinf(x: NDArrayCore): NDArray {
  return up(core.isposinf(x));
}

/** Element-wise test for complex values */
export function iscomplex(x: NDArrayCore): NDArray {
  return up(core.iscomplex(x));
}

/** Element-wise test for real values */
export function isreal(x: NDArrayCore): NDArray {
  return up(core.isreal(x));
}

/** Return real array if imaginary part is negligible */
export function real_if_close(x: NDArrayCore, tol?: number): NDArray {
  return up(core.real_if_close(x, tol));
}

/** Return sign of x1 with magnitude of x2 */
export function copysign(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.copysign(x1, x2));
}

/** Test for negative sign bit */
export function signbit(x: NDArrayCore): NDArray {
  return up(core.signbit(x));
}

/** Next floating-point value toward x2 */
export function nextafter(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.nextafter(x1, x2));
}

/** Spacing between x and nearest adjacent number */
export function spacing(x: NDArrayCore): NDArray {
  return up(core.spacing(x));
}

/** Element-wise greater than comparison */
export function greater(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.greater(x1, x2));
}

/** Element-wise greater than or equal comparison */
export function greater_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.greater_equal(x1, x2));
}

/** Element-wise less than comparison */
export function less(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.less(x1, x2));
}

/** Element-wise less than or equal comparison */
export function less_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.less_equal(x1, x2));
}

/** Element-wise equality comparison */
export function equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.equal(x1, x2));
}

/** Element-wise inequality comparison */
export function not_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.not_equal(x1, x2));
}

/** Element-wise close comparison with tolerance */
export function isclose(
  a: NDArrayCore,
  b: NDArrayCore | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): NDArray {
  return up(core.isclose(a, b, rtol, atol));
}

/**
 * Find the coefficients of a polynomial with given roots
 */
export function poly(seq_of_zeros: NDArrayCore | number[]): NDArray {
  return up(core.poly(seq_of_zeros));
}

/**
 * Add two polynomials
 */
export function polyadd(a1: NDArrayCore | number[], a2: NDArrayCore | number[]): NDArray {
  return up(core.polyadd(a1, a2));
}

/**
 * Differentiate a polynomial
 */
export function polyder(p: NDArrayCore | number[], m: number = 1): NDArray {
  return up(core.polyder(p, m));
}

/**
 * Divide two polynomials
 */
export function polydiv(u: NDArrayCore | number[], v: NDArrayCore | number[]): [NDArray, NDArray] {
  const r = core.polydiv(u, v);
  return [up(r[0]), up(r[1])] as [NDArray, NDArray];
}

/**
 * Least squares polynomial fit
 */
export function polyfit(x: NDArrayCore, y: NDArrayCore, deg: number): NDArray {
  return up(core.polyfit(x, y, deg));
}

/**
 * Integrate a polynomial
 */
export function polyint(
  p: NDArrayCore | number[],
  m: number = 1,
  k: number | number[] = 0
): NDArray {
  return up(core.polyint(p, m, k));
}

/**
 * Multiply two polynomials
 */
export function polymul(a1: NDArrayCore | number[], a2: NDArrayCore | number[]): NDArray {
  return up(core.polymul(a1, a2));
}

/**
 * Subtract two polynomials
 */
export function polysub(a1: NDArrayCore | number[], a2: NDArrayCore | number[]): NDArray {
  return up(core.polysub(a1, a2));
}

/**
 * Evaluate a polynomial at given points
 */
export function polyval(
  p: NDArrayCore | number[],
  x: NDArrayCore | number | number[]
): NDArray | number {
  const r = core.polyval(p, x);
  return r instanceof NDArrayCore ? up(r) : r;
}

/**
 * Find the roots of a polynomial.
 *
 * Uses the companion matrix eigenvalue method (same as NumPy).
 * Always returns a complex128 NDArrayCore.
 */
export function roots(p: NDArrayCore | number[]): NDArray {
  return up(core.roots(p));
}

/** Sum of array elements */
export function sum(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | bigint | Complex {
  const r = core.sum(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Mean of array elements */
export function mean(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.mean(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Product of array elements */
export function prod(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | bigint | Complex {
  const r = core.prod(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Maximum of array elements */
export function max(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | number | Complex {
  const r = core.max(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Minimum of array elements */
export function min(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | number | Complex {
  const r = core.min(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Peak-to-peak (max - min) */
export function ptp(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | number | Complex {
  const r = core.ptp(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Index of minimum value */
export function argmin(a: NDArrayCore, axis?: number): NDArray | number {
  const r = core.argmin(a, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Index of maximum value */
export function argmax(a: NDArrayCore, axis?: number): NDArray | number {
  const r = core.argmax(a, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Variance of array elements */
export function variance(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.variance(a, axis, ddof, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Standard deviation of array elements */
export function std(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.std(a, axis, ddof, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Median of array elements */
export function median(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | number {
  const r = core.median(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Percentile of array elements */
export function percentile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.percentile(a, q, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Quantile of array elements */
export function quantile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.quantile(a, q, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Weighted average */
export function average(
  a: NDArrayCore,
  axis?: number,
  weights?: NDArrayCore,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.average(a, axis, weights, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Test if all elements are true */
export function all(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | boolean {
  const r = core.all(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Test if any element is true */
export function any(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | boolean {
  const r = core.any(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Cumulative sum */
export function cumsum(a: NDArrayCore, axis?: number): NDArray {
  return up(core.cumsum(a, axis));
}

/** Cumulative product */
export function cumprod(a: NDArrayCore, axis?: number): NDArray {
  return up(core.cumprod(a, axis));
}

/** Sum ignoring NaN */
export function nansum(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.nansum(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Product ignoring NaN */
export function nanprod(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.nanprod(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Mean ignoring NaN */
export function nanmean(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.nanmean(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Variance ignoring NaN */
export function nanvar(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.nanvar(a, axis, ddof, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Standard deviation ignoring NaN */
export function nanstd(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.nanstd(a, axis, ddof, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Minimum ignoring NaN */
export function nanmin(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.nanmin(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Maximum ignoring NaN */
export function nanmax(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArray | number | Complex {
  const r = core.nanmax(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Index of minimum ignoring NaN */
export function nanargmin(a: NDArrayCore, axis?: number): NDArray | number {
  const r = core.nanargmin(a, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Index of maximum ignoring NaN */
export function nanargmax(a: NDArrayCore, axis?: number): NDArray | number {
  const r = core.nanargmax(a, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Cumulative sum ignoring NaN */
export function nancumsum(a: NDArrayCore, axis?: number): NDArray {
  return up(core.nancumsum(a, axis));
}

/** Cumulative product ignoring NaN */
export function nancumprod(a: NDArrayCore, axis?: number): NDArray {
  return up(core.nancumprod(a, axis));
}

/** Median ignoring NaN */
export function nanmedian(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArray | number {
  const r = core.nanmedian(a, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Quantile ignoring NaN */
export function nanquantile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.nanquantile(a, q, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Percentile ignoring NaN */
export function nanpercentile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArray | number {
  const r = core.nanpercentile(a, q, axis, keepdims);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Round to given decimals */
export function around(a: NDArrayCore, decimals: number = 0): NDArray {
  return up(core.around(a, decimals));
}

/** Round (same as around) */
export function round(a: NDArrayCore, decimals: number = 0): NDArray {
  return up(core.round(a, decimals));
}

/** Ceiling */
export function ceil(x: NDArrayCore): NDArray {
  return up(core.ceil(x));
}

/** Round toward zero */
export function fix(x: NDArrayCore): NDArray {
  return up(core.fix(x));
}

/** Floor */
export function floor(x: NDArrayCore): NDArray {
  return up(core.floor(x));
}

/** Round to nearest integer */
export function rint(x: NDArrayCore): NDArray {
  return up(core.rint(x));
}

/** Truncate toward zero */
export function trunc(x: NDArrayCore): NDArray {
  return up(core.trunc(x));
}

/** Test if elements in ar1 are in ar2 (deprecated, use isin) */
export function in1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.in1d(ar1, ar2));
}

/** Find intersection of two arrays */
export function intersect1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.intersect1d(ar1, ar2));
}

/** Test if elements are in test elements */
export function isin(element: NDArrayCore, testElements: NDArrayCore): NDArray {
  return up(core.isin(element, testElements));
}

/** Find set difference (elements in ar1 not in ar2) */
export function setdiff1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.setdiff1d(ar1, ar2));
}

/** Find symmetric difference (elements in either but not both) */
export function setxor1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.setxor1d(ar1, ar2));
}

/** Find union of two arrays */
export function union1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.union1d(ar1, ar2));
}

/** Trim leading/trailing zeros */
export function trim_zeros(filt: NDArrayCore, trim?: 'f' | 'b' | 'fb'): NDArray {
  return up(core.trim_zeros(filt, trim));
}

/** Return unique values (Array API) */
export function unique_values(x: NDArrayCore): NDArray {
  return up(core.unique_values(x));
}

/**
 * Append values to the end of an array
 */
export function append(
  arr: NDArrayCore,
  values: NDArrayCore | number | number[],
  axis?: number
): NDArray {
  return up(core.append(arr, values, axis));
}

/**
 * Delete elements from an array
 */
export function delete_(arr: NDArrayCore, obj: number | number[], axis?: number): NDArray {
  return up(core.delete(arr, obj, axis));
}

/**
 * Insert values along an axis
 */
export function insert(
  arr: NDArrayCore,
  obj: number | number[],
  values: NDArrayCore | number | number[],
  axis?: number
): NDArray {
  return up(core.insert(arr, obj, values, axis));
}

/**
 * Pad an array
 */
export function pad(
  arr: NDArrayCore,
  pad_width: number | [number, number] | [number, number][],
  mode:
    | 'constant'
    | 'edge'
    | 'linear_ramp'
    | 'maximum'
    | 'mean'
    | 'median'
    | 'minimum'
    | 'reflect'
    | 'symmetric'
    | 'wrap'
    | 'empty' = 'constant',
  constant_values: number = 0
): NDArray {
  return up(core.pad(arr, pad_width, mode, constant_values));
}

/** Reshape array without changing data */
export function reshape(a: NDArrayCore, newShape: number[]): NDArray {
  return up(core.reshape(a, newShape));
}

/** Flatten array to 1D */
export function flatten(a: NDArrayCore): NDArray {
  return up(core.flatten(a));
}

/** Return contiguous flattened array */
export function ravel(a: NDArrayCore): NDArray {
  return up(core.ravel(a));
}

/** Remove single-dimensional entries from shape - returns a view */
export function squeeze(a: NDArrayCore, axis?: number): NDArray {
  return up(core.squeeze(a, axis));
}

/** Expand array dimensions - returns a view */
export function expand_dims(a: NDArrayCore, axis: number): NDArray {
  return up(core.expand_dims(a, axis));
}

/** Interchange two axes - returns a view */
export function swapaxes(a: NDArrayCore, axis1: number, axis2: number): NDArray {
  return up(core.swapaxes(a, axis1, axis2));
}

/** Move axis to new position - returns a view */
export function moveaxis(
  a: NDArrayCore,
  source: number | number[],
  destination: number | number[]
): NDArray {
  return up(core.moveaxis(a, source, destination));
}

/** Roll axis to given position - returns a view */
export function rollaxis(a: NDArrayCore, axis: number, start: number = 0): NDArray {
  return up(core.rollaxis(a, axis, start));
}

/** Join arrays along an existing axis */
export function concatenate(arrays: NDArrayCore[], axis: number = 0): NDArray {
  return up(core.concatenate(arrays, axis));
}

/** Join arrays along a new axis */
export function stack(arrays: NDArrayCore[], axis: number = 0): NDArray {
  return up(core.stack(arrays, axis));
}

/** Stack arrays vertically (row-wise) */
export function vstack(arrays: NDArrayCore[]): NDArray {
  return up(core.vstack(arrays));
}

/** Stack arrays horizontally (column-wise) */
export function hstack(arrays: NDArrayCore[]): NDArray {
  return up(core.hstack(arrays));
}

/** Stack arrays along third axis (depth-wise) */
export function dstack(arrays: NDArrayCore[]): NDArray {
  return up(core.dstack(arrays));
}

/** Concatenate (alias) */
export function concat(arrays: NDArrayCore[], axis: number = 0): NDArray {
  return up(core.concat(arrays, axis));
}

/** Stack 1D arrays as columns */
export function column_stack(arrays: NDArrayCore[]): NDArray {
  return up(core.column_stack(arrays));
}

/** Assemble arrays from nested sequences of blocks */
export function block(arrays: NDArrayCore[]): NDArray {
  return up(core.block(arrays));
}

/** Unstack array into list of arrays */
export function unstack(a: NDArrayCore, axis: number = 0): NDArray[] {
  return core.unstack(a, axis).map(up);
}

/** Tile array by given repetitions */
export function tile(a: NDArrayCore, reps: number | number[]): NDArray {
  return up(core.tile(a, reps));
}

/** Repeat elements of array */
export function repeat(a: NDArrayCore, repeats: number | number[], axis?: number): NDArray {
  return up(core.repeat(a, repeats, axis));
}

/** Reverse order of elements along axis */
export function flip(m: NDArrayCore, axis?: number | number[]): NDArray {
  return up(core.flip(m, axis));
}

/** Flip array left/right */
export function fliplr(m: NDArrayCore): NDArray {
  return up(core.fliplr(m));
}

/** Flip array up/down */
export function flipud(m: NDArrayCore): NDArray {
  return up(core.flipud(m));
}

/** Rotate array by 90 degrees */
export function rot90(m: NDArrayCore, k: number = 1, axes: [number, number] = [0, 1]): NDArray {
  return up(core.rot90(m, k, axes));
}

/** Roll array elements along axis */
export function roll(a: NDArrayCore, shift: number | number[], axis?: number | number[]): NDArray {
  return up(core.roll(a, shift, axis));
}

/** Resize array to new shape */
export function resize(a: NDArrayCore, new_shape: number[]): NDArray {
  return up(core.resize(a, new_shape));
}

/** Sort array along axis */
export function sort(a: NDArrayCore, axis: number = -1): NDArray {
  return up(core.sort(a, axis));
}

/** Indices that would sort array */
export function argsort(a: NDArrayCore, axis: number = -1): NDArray {
  return up(core.argsort(a, axis));
}

/** Indirect stable sort on multiple keys */
export function lexsort(keys: NDArrayCore[]): NDArray {
  return up(core.lexsort(keys));
}

/** Partially sort array */
export function partition(a: NDArrayCore, kth: number, axis: number = -1): NDArray {
  return up(core.partition(a, kth, axis));
}

/** Indices that would partially sort array */
export function argpartition(a: NDArrayCore, kth: number, axis: number = -1): NDArray {
  return up(core.argpartition(a, kth, axis));
}

/** Sort complex array */
export function sort_complex(a: NDArrayCore): NDArray {
  return up(core.sort_complex(a));
}

/** Indices of non-zero elements */
export function nonzero(a: NDArrayCore): NDArray[] {
  return core.nonzero(a).map(up);
}

/** Indices where condition is True */
export function argwhere(a: NDArrayCore): NDArray {
  return up(core.argwhere(a));
}

/** Indices of non-zero elements in flattened array */
export function flatnonzero(a: NDArrayCore): NDArray {
  return up(core.flatnonzero(a));
}

/** Extract elements where condition is True */
export function extract(condition: NDArrayCore, a: NDArrayCore): NDArray {
  return up(core.extract(condition, a));
}

/** Count occurrences of values */
export function bincount(x: NDArrayCore, weights?: NDArrayCore, minlength?: number): NDArray {
  return up(core.bincount(x, weights, minlength));
}

/** Digitize values into bins */
export function digitize(x: NDArrayCore, bins: NDArrayCore, right?: boolean): NDArray {
  return up(core.digitize(x, bins, right));
}

/** Cross-correlation */
export function correlate(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full'
): NDArray {
  return up(core.correlate(a, v, mode));
}

/** Convolution */
export function convolve(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full'
): NDArray {
  return up(core.convolve(a, v, mode));
}

/** Covariance matrix */
export function cov(
  m: NDArrayCore,
  y?: NDArrayCore,
  rowvar?: boolean,
  bias?: boolean,
  ddof?: number
): NDArray {
  return up(core.cov(m, y, rowvar, bias, ddof));
}

/** Correlation coefficients */
export function corrcoef(x: NDArrayCore, y?: NDArrayCore, rowvar?: boolean): NDArray {
  return up(core.corrcoef(x, y, rowvar));
}

/** Integrate using trapezoidal rule */
export function trapezoid(
  y: NDArrayCore,
  x?: NDArrayCore,
  dx?: number,
  axis?: number
): NDArray | number {
  const r = core.trapezoid(y, x, dx, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Sine of array elements */
export function sin(x: NDArrayCore): NDArray {
  return up(core.sin(x));
}

/** Cosine of array elements */
export function cos(x: NDArrayCore): NDArray {
  return up(core.cos(x));
}

/** Tangent of array elements */
export function tan(x: NDArrayCore): NDArray {
  return up(core.tan(x));
}

/** Inverse sine */
export function arcsin(x: NDArrayCore): NDArray {
  return up(core.arcsin(x));
}

/** Inverse cosine */
export function arccos(x: NDArrayCore): NDArray {
  return up(core.arccos(x));
}

/** Inverse tangent */
export function arctan(x: NDArrayCore): NDArray {
  return up(core.arctan(x));
}

/** Two-argument inverse tangent */
export function arctan2(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.arctan2(x1, x2));
}

/** Hypotenuse (sqrt(x1^2 + x2^2)) */
export function hypot(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.hypot(x1, x2));
}

/** Convert radians to degrees */
export function degrees(x: NDArrayCore): NDArray {
  return up(core.degrees(x));
}

/** Convert degrees to radians */
export function radians(x: NDArrayCore): NDArray {
  return up(core.radians(x));
}

/** Convert degrees to radians (alias) */
export function deg2rad(x: NDArrayCore): NDArray {
  return up(core.deg2rad(x));
}

/** Convert radians to degrees (alias) */
export function rad2deg(x: NDArrayCore): NDArray {
  return up(core.rad2deg(x));
}

/** Hyperbolic sine */
export function sinh(x: NDArrayCore): NDArray {
  return up(core.sinh(x));
}

/** Hyperbolic cosine */
export function cosh(x: NDArrayCore): NDArray {
  return up(core.cosh(x));
}

/** Hyperbolic tangent */
export function tanh(x: NDArrayCore): NDArray {
  return up(core.tanh(x));
}

/** Inverse hyperbolic sine */
export function arcsinh(x: NDArrayCore): NDArray {
  return up(core.arcsinh(x));
}

/** Inverse hyperbolic cosine */
export function arccosh(x: NDArrayCore): NDArray {
  return up(core.arccosh(x));
}

/** Inverse hyperbolic tangent */
export function arctanh(x: NDArrayCore): NDArray {
  return up(core.arctanh(x));
}

// ============================================================
// Re-exported Functions (no wrapping needed)
// ============================================================

export {
  broadcast_arrays,
  broadcast_shapes,
  put,
  put_along_axis,
  place,
  putmask,
  copyto,
  ix_,
  fill_diagonal,
  array_equal,
  array_equiv,
  may_share_memory,
  shares_memory,
  geterr,
  seterr,
  pow,
  abs,
  true_divide,
  conjugate,
  array2string,
  array_repr,
  array_str,
  set_printoptions,
  get_printoptions,
  printoptions,
  format_float_positional,
  format_float_scientific,
  base_repr,
  binary_repr,
  gradient,
  trace,
  einsum,
  einsum_path,
  vdot,
  linalg,
  iscomplexobj,
  isrealobj,
  isfortran,
  isscalar,
  iterable,
  isdtype,
  promote_types,
  allclose,
  amax,
  amin,
  var_,
  cumulative_sum,
  cumulative_prod,
  round_,
  unique,
  unique_all,
  unique_counts,
  unique_inverse,
  split,
  array_split,
  vsplit,
  hsplit,
  dsplit,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  row_stack,
  where,
  searchsorted,
  count_nonzero,
  histogram,
  histogram2d,
  histogramdd,
  histogram_bin_edges,
  asin,
  acos,
  atan,
  atan2,
  asinh,
  acosh,
  atanh,
  can_cast,
  common_type,
  result_type,
  min_scalar_type,
  issubdtype,
  typename,
  mintypecode,
  ndim,
  shape,
  size,
  item,
  tolist,
  tobytes,
  byteswap,
  view,
  tofile,
  fill,
} from '../core';
