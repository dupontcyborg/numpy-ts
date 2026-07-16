// AUTO-GENERATED - DO NOT EDIT
// Run `pnpm run generate` to regenerate this file from core/ modules

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

import { Complex } from '../common/complex';
import {
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  isComplexDType,
  promoteDTypes,
  type Scalar,
} from '../common/dtype';
import type {
  Abs,
  Angle,
  BoolArith,
  ComplexComponent,
  Divide,
  FloatPower,
  MathBinary,
  MathResult,
  Power,
  Promote,
  ReductionAccum,
  RoundResult,
  StdVar,
  TrueDivide,
} from '../common/dtype-promotion';
import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import * as core from '../core';
import type { ReductionOpts } from '../core/reduction';
import type { NestedNDArrays } from '../core/shape';
import type { PadValueArg, PadWidthArg } from '../core/shape-extra';
import type { ArrayLike, DType, TypedArray } from '../core/types';
import { NDArray } from './ndarray';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
// If already an NDArray, return as-is to preserve identity
// Also preserves base reference for views
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

export { Complex } from '../common/complex';
export { NDArrayCore } from '../common/ndarray-core';
export type { NDIndex } from '../core/advanced';
// Re-export types
export type { DType, TypedArray } from '../core/types';
export { meshgrid, NDArray } from './ndarray';

// ============================================================
// Wrapped Functions (return NDArray)
// ============================================================

/** Broadcast array to a new shape - returns a view */
export function broadcast_to<D extends DType>(a: NDArrayCore<D>, shape: number[]): NDArray<D> {
  return up(core.broadcast_to(a, shape)) as NDArray<D>;
}

export function take<D extends DType>(
  a: NDArrayCore<D>,
  indices: ArrayLike,
  axis?: number,
): NDArray<D> {
  return up(core.take(a, indices, axis)) as NDArray<D>;
}

export function take_along_axis<D extends DType>(
  arr: NDArrayCore<D>,
  indices: NDArrayCore,
  axis: number,
): NDArray<D> {
  return up(core.take_along_axis(arr, indices, axis)) as NDArray<D>;
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
  axis: number = 0,
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
  condlist: ArrayLike[],
  choicelist: ArrayLike[],
  defaultVal: ArrayLike = 0,
): NDArray {
  return up(core.select(condlist, choicelist, defaultVal));
}

export function indices(dimensions: number[], dtype: string = 'float64'): NDArray {
  return up(core.indices(dimensions, dtype));
}

export function ravel_multi_index<D extends DType>(
  multi_index: NDArrayCore<D>[],
  dims: number[],
  mode: 'raise' | 'wrap' | 'clip' = 'raise',
): NDArray<'float64'> {
  return up(core.ravel_multi_index(multi_index, dims, mode)) as NDArray<'float64'>;
}

export function unravel_index(
  indices: NDArrayCore | number,
  shape: number[],
): NDArray<'float64'>[] {
  return core.unravel_index(indices, shape).map(up) as NDArray<'float64'>[];
}

export function diag_indices(n: number, ndim: number = 2): NDArray<'float64'>[] {
  return core.diag_indices(n, ndim).map(up) as NDArray<'float64'>[];
}

export function diag_indices_from(a: NDArrayCore): NDArray<'float64'>[] {
  return core.diag_indices_from(a).map(up) as NDArray<'float64'>[];
}

export function tril_indices(n: number, k: number = 0, m?: number): NDArray<'float64'>[] {
  return core.tril_indices(n, k, m).map(up) as NDArray<'float64'>[];
}

export function tril_indices_from(a: NDArrayCore, k: number = 0): NDArray<'float64'>[] {
  return core.tril_indices_from(a, k).map(up) as NDArray<'float64'>[];
}

export function triu_indices(n: number, k: number = 0, m?: number): NDArray<'float64'>[] {
  return core.triu_indices(n, k, m).map(up) as NDArray<'float64'>[];
}

export function triu_indices_from(a: NDArrayCore, k: number = 0): NDArray<'float64'>[] {
  return core.triu_indices_from(a, k).map(up) as NDArray<'float64'>[];
}

export function mask_indices(
  n: number,
  mask_func: (m: NDArrayCore, k: number) => NDArrayCore,
  k: number = 0,
): NDArray<'float64'>[] {
  return core.mask_indices(n, mask_func, k).map(up) as NDArray<'float64'>[];
}

export function apply_along_axis(
  func1d: (arr: NDArrayCore, ...args: unknown[]) => NDArrayCore | number,
  axis: number,
  arr: NDArrayCore,
  ...args: unknown[]
): NDArray {
  const wrappedFunc1d = (arr: NDArrayCore, ...passed: unknown[]): NDArrayCore | number => {
    return func1d(up(arr), ...passed);
  };
  return up(core.apply_along_axis(wrappedFunc1d, axis, arr, ...args));
}

export function apply_over_axes(
  func: (arr: NDArrayCore, axis: number) => NDArrayCore,
  a: NDArrayCore,
  axes: number[],
): NDArray {
  const wrappedFunc = (arr: NDArrayCore, axis: number): NDArrayCore => {
    return func(up(arr), axis);
  };
  return up(core.apply_over_axes(wrappedFunc, a, axes));
}

/**
 * Vectorized multi-dimensional indexing (dask.vindex semantics).
 *
 * Integer-array subspace dimensions come first in the output, followed by
 * slice dimensions in their original order.
 */
export function vindex(
  a: NDArrayCore,
  ...indices: (number | string | number[] | NDArrayCore)[]
): NDArray {
  return up(core.vindex(a, ...indices));
}

/** Add arguments element-wise */
export function add<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function add<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function add(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.add(x1, x2));
}

/** Subtract arguments element-wise */
export function subtract<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function subtract<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function subtract(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.subtract(x1, x2));
}

/** Multiply arguments element-wise */
export function multiply<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function multiply<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function multiply(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.multiply(x1, x2));
}

/** Square root of array elements */
export function sqrt<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.sqrt(x)) as NDArray<MathResult<D>>;
}

/** Element-wise power */
export function power<A extends DType, B extends DType>(
  x: NDArrayCore<A>,
  exponent: NDArrayCore<B>,
): NDArray<Power<A, B>>;
export function power<A extends DType>(x: NDArrayCore<A>, exponent: number): NDArray<A>;
export function power(x: NDArrayCore, exponent: NDArrayCore | number): NDArray {
  return up(core.power(x, exponent));
}

/** Exponential (e^x) */
export function exp<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.exp(x)) as NDArray<MathResult<D>>;
}

/** 2^x */
export function exp2<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.exp2(x)) as NDArray<MathResult<D>>;
}

/** exp(x) - 1 */
export function expm1<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.expm1(x)) as NDArray<MathResult<D>>;
}

/** Natural logarithm */
export function log<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.log(x)) as NDArray<MathResult<D>>;
}

/** Base-2 logarithm */
export function log2<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.log2(x)) as NDArray<MathResult<D>>;
}

/** Base-10 logarithm */
export function log10<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.log10(x)) as NDArray<MathResult<D>>;
}

/** log(1 + x) */
export function log1p<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.log1p(x)) as NDArray<MathResult<D>>;
}

/** log(exp(x1) + exp(x2)) */
export function logaddexp<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function logaddexp<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function logaddexp(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logaddexp(x1, x2));
}

/** log2(2^x1 + 2^x2) */
export function logaddexp2<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function logaddexp2<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function logaddexp2(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logaddexp2(x1, x2));
}

/** Absolute value */
export function absolute<D extends DType>(x: NDArrayCore<D>): NDArray<Abs<D>> {
  return up(core.absolute(x)) as NDArray<Abs<D>>;
}

/** Numerical negative */
export function negative<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.negative(x)) as NDArray<D>;
}

/** Element-wise sign */
export function sign<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.sign(x)) as NDArray<D>;
}

/** Element-wise modulo */
export function mod<A extends DType, B extends DType>(
  x: NDArrayCore<A>,
  divisor: NDArrayCore<B>,
): NDArray<Power<A, B>>;
export function mod<A extends DType>(x: NDArrayCore<A>, divisor: number): NDArray<A>;
export function mod(x: NDArrayCore, divisor: NDArrayCore | number): NDArray {
  return up(core.mod(x, divisor));
}

/** Element-wise division */
export function divide<A extends DType, B extends DType>(
  x: NDArrayCore<A>,
  divisor: NDArrayCore<B>,
): NDArray<Divide<A, B>>;
export function divide<A extends DType>(x: NDArrayCore<A>, divisor: number): NDArray<TrueDivide<A>>;
export function divide(x: NDArrayCore, divisor: NDArrayCore | number): NDArray {
  return up(core.divide(x, divisor));
}

/** Element-wise floor division */
export function floor_divide<A extends DType, B extends DType>(
  x: NDArrayCore<A>,
  divisor: NDArrayCore<B>,
): NDArray<Power<A, B>>;
export function floor_divide<A extends DType>(x: NDArrayCore<A>, divisor: number): NDArray<A>;
export function floor_divide(x: NDArrayCore, divisor: NDArrayCore | number): NDArray {
  return up(core.floor_divide(x, divisor));
}

/** Numerical positive (returns copy) */
export function positive<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.positive(x)) as NDArray<D>;
}

/** Reciprocal (1/x) */
export function reciprocal<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.reciprocal(x)) as NDArray<D>;
}

/** Cube root */
export function cbrt<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.cbrt(x)) as NDArray<MathResult<D>>;
}

/** Absolute value (float) */
export function fabs<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.fabs(x)) as NDArray<MathResult<D>>;
}

/** Element-wise divmod (quotient and remainder) */
export function divmod<A extends DType, B extends DType>(
  x: NDArrayCore<A>,
  y: NDArrayCore<B>,
): [NDArray<Power<A, B>>, NDArray<Power<A, B>>];
export function divmod<A extends DType>(x: NDArrayCore<A>, y: number): [NDArray<A>, NDArray<A>];
export function divmod(x: NDArrayCore, y: NDArrayCore | number): [NDArray, NDArray] {
  const r = core.divmod(x, y);
  return [up(r[0]), up(r[1])] as [NDArray, NDArray];
}

/** Element-wise square */
export function square<D extends DType>(x: NDArrayCore<D>): NDArray<BoolArith<D>> {
  return up(core.square(x)) as NDArray<BoolArith<D>>;
}

/** Element-wise remainder */
export function remainder<A extends DType, B extends DType>(
  x: NDArrayCore<A>,
  y: NDArrayCore<B>,
): NDArray<Power<A, B>>;
export function remainder<A extends DType>(x: NDArrayCore<A>, y: number): NDArray<A>;
export function remainder(x: NDArrayCore, y: NDArrayCore | number): NDArray {
  return up(core.remainder(x, y));
}

/** Heaviside step function */
export function heaviside<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function heaviside<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function heaviside(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.heaviside(x1, x2));
}

/** Float power (always returns float) */
export function float_power<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<FloatPower<A, B>>;
export function float_power<A extends DType>(
  x1: NDArrayCore<A>,
  x2: number,
): NDArray<Promote<A, 'float64'>>;
export function float_power(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.float_power(x1, x2));
}

/** C-style fmod */
export function fmod<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function fmod<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function fmod(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.fmod(x1, x2));
}

/** Decompose into mantissa and exponent */
export function frexp<D extends DType>(
  x: NDArrayCore<D>,
): [NDArray<MathResult<D>>, NDArray<'int32'>] {
  const r = core.frexp(x);
  return [up(r[0]), up(r[1])] as unknown as [NDArray<MathResult<D>>, NDArray<'int32'>];
}

/** Greatest common divisor */
export function gcd<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function gcd<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function gcd(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.gcd(x1, x2));
}

/** Least common multiple */
export function lcm<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function lcm<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function lcm(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.lcm(x1, x2));
}

/** Compose from mantissa and exponent */
export function ldexp<D extends DType>(x1: NDArrayCore<D>, x2: NDArrayCore | number): NDArray<D> {
  return up(core.ldexp(x1, x2)) as NDArray<D>;
}

/** Decompose into fractional and integer parts */
export function modf<D extends DType>(
  x: NDArrayCore<D>,
): [NDArray<MathResult<D>>, NDArray<MathResult<D>>] {
  const r = core.modf(x);
  return [up(r[0]), up(r[1])] as unknown as [NDArray<MathResult<D>>, NDArray<MathResult<D>>];
}

/** Clip values to a range */
export function clip(
  a: NDArrayCore,
  a_min: NDArrayCore | number | null,
  a_max: NDArrayCore | number | null,
): NDArray {
  return up(core.clip(a, a_min, a_max));
}

/** Element-wise maximum */
export function maximum<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function maximum<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function maximum(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.maximum(x1, x2));
}

/** Element-wise minimum */
export function minimum<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function minimum<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function minimum(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.minimum(x1, x2));
}

/** Element-wise maximum (ignores NaN) */
export function fmax<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function fmax<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function fmax(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.fmax(x1, x2));
}

/** Element-wise minimum (ignores NaN) */
export function fmin<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function fmin<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function fmin(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.fmin(x1, x2));
}

/** Replace NaN/Inf with numbers */
export function nan_to_num<D extends DType>(
  x: NDArrayCore<D>,
  nan: number = 0.0,
  posinf?: number,
  neginf?: number,
): NDArray<D> {
  return up(core.nan_to_num(x, nan, posinf, neginf)) as NDArray<D>;
}

/** 1D linear interpolation. If `period` is given, x and xp are normalized into
 *  `[0, period)` and the curve wraps at the boundary; `left`/`right` are ignored. */
export function interp(
  x: NDArrayCore,
  xp: NDArrayCore,
  fp: NDArrayCore,
  left?: number,
  right?: number,
  period?: number,
): NDArray {
  return up(core.interp(x, xp, fp, left, right, period));
}

/** Unwrap phase angles */
export function unwrap<D extends DType>(
  p: NDArrayCore<D>,
  discont: number = Math.PI,
  axis: number = -1,
  period: number = 2 * Math.PI,
): NDArray<TrueDivide<D>> {
  return up(core.unwrap(p, discont, axis, period)) as NDArray<TrueDivide<D>>;
}

/** Sinc function */
export function sinc<D extends DType>(x: NDArrayCore<D>): NDArray<TrueDivide<D>> {
  return up(core.sinc(x)) as NDArray<TrueDivide<D>>;
}

/** Modified Bessel function of the first kind, order 0 */
export function i0<D extends DType>(x: NDArrayCore<D>): NDArray<TrueDivide<D>> {
  return up(core.i0(x)) as NDArray<TrueDivide<D>>;
}

/** Bitwise AND */
export function bitwise_and<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function bitwise_and<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function bitwise_and(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_and(x1, x2));
}

/** Bitwise OR */
export function bitwise_or<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function bitwise_or<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function bitwise_or(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_or(x1, x2));
}

/** Bitwise XOR */
export function bitwise_xor<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function bitwise_xor<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function bitwise_xor(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.bitwise_xor(x1, x2));
}

/** Bitwise NOT */
export function bitwise_not<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.bitwise_not(x)) as NDArray<D>;
}

/** Bitwise inversion (alias for bitwise_not) */
export function invert<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.invert(x)) as NDArray<D>;
}

/** Left shift */
export function left_shift<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function left_shift<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
export function left_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.left_shift(x1, x2));
}

/** Right shift */
export function right_shift<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function right_shift<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<A>;
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
  bitorder?: 'big' | 'little',
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
export function real<D extends DType>(x: NDArrayCore<D>): NDArray<ComplexComponent<D>> {
  return up(core.real(x)) as NDArray<ComplexComponent<D>>;
}

/** Extract imaginary part of array */
export function imag<D extends DType>(x: NDArrayCore<D>): NDArray<ComplexComponent<D>> {
  return up(core.imag(x)) as NDArray<ComplexComponent<D>>;
}

/** Complex conjugate */
export function conj<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.conj(x)) as NDArray<D>;
}

/** Phase angle */
export function angle<D extends DType>(x: NDArrayCore<D>, deg?: boolean): NDArray<Angle<D>> {
  return up(core.angle(x, deg)) as NDArray<Angle<D>>;
}

/**
 * Create array of zeros
 */
export function zeros<D extends DType = 'float64'>(
  shape: number[],
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.zeros(shape, dtype)) as NDArray<D>;
}

/**
 * Create array of ones
 */
export function ones<D extends DType = 'float64'>(
  shape: number[],
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.ones(shape, dtype)) as NDArray<D>;
}

/**
 * Create an uninitialized array
 */
export function empty<D extends DType = 'float64'>(
  shape: number[],
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.empty(shape, dtype)) as NDArray<D>;
}

/**
 * Create array filled with a constant value
 */
export function full<D extends DType = DType>(
  shape: number[],
  fill_value: number | bigint | boolean,
  dtype?: D,
): NDArray<D> {
  return up(core.full(shape, fill_value, dtype)) as NDArray<D>;
}

/**
 * Create array from nested JavaScript arrays
 */
export function array<D extends DType = DType>(data: unknown, dtype?: D): NDArray<D> {
  return up(core.array(data, dtype)) as NDArray<D>;
}

/**
 * Create array with evenly spaced values within a given interval
 */
export function arange<D extends DType = DType>(
  start: number,
  stop?: number,
  step: number = 1,
  dtype?: D,
): NDArray<D> {
  return up(core.arange(start, stop, step, dtype)) as NDArray<D>;
}

/**
 * Create array with evenly spaced values over a specified interval
 */
export function linspace<D extends DType = 'float64'>(
  start: number,
  stop: number,
  num: number = 50,
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.linspace(start, stop, num, dtype)) as NDArray<D>;
}

/**
 * Create array with logarithmically spaced values
 */
export function logspace<D extends DType = 'float64'>(
  start: number,
  stop: number,
  num: number = 50,
  base: number = 10.0,
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.logspace(start, stop, num, base, dtype)) as NDArray<D>;
}

/**
 * Create array with geometrically spaced values
 */
export function geomspace<D extends DType = 'float64'>(
  start: number,
  stop: number,
  num: number = 50,
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.geomspace(start, stop, num, dtype)) as NDArray<D>;
}

/**
 * Create identity matrix
 */
export function eye<D extends DType = 'float64'>(
  n: number,
  m?: number,
  k: number = 0,
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.eye(n, m, k, dtype)) as NDArray<D>;
}

/**
 * Create a square identity matrix
 */
export function identity<D extends DType = 'float64'>(
  n: number,
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.identity(n, dtype)) as NDArray<D>;
}

/**
 * Convert input to an ndarray
 */
export function asarray<D extends DType = DType>(a: NDArrayCore | unknown, dtype?: D): NDArray<D> {
  return up(core.asarray(a, dtype)) as NDArray<D>;
}

/**
 * Return array of zeros with the same shape and dtype as input
 */
export function zeros_like<D extends DType = DType>(a: NDArrayCore, dtype?: D): NDArray<D> {
  return up(core.zeros_like(a, dtype)) as NDArray<D>;
}

/**
 * Return array of ones with the same shape and dtype as input
 */
export function ones_like<D extends DType = DType>(a: NDArrayCore, dtype?: D): NDArray<D> {
  return up(core.ones_like(a, dtype)) as NDArray<D>;
}

/**
 * Return empty array with the same shape and dtype as input
 */
export function empty_like<D extends DType = DType>(a: NDArrayCore, dtype?: D): NDArray<D> {
  return up(core.empty_like(a, dtype)) as NDArray<D>;
}

/**
 * Return array filled with value, same shape and dtype as input
 */
export function full_like<D extends DType = DType>(
  a: NDArrayCore,
  fill_value: number | bigint | boolean,
  dtype?: D,
): NDArray<D> {
  return up(core.full_like(a, fill_value, dtype)) as NDArray<D>;
}

/**
 * Deep copy of array
 */
export function copy<D extends DType>(a: NDArrayCore<D>): NDArray<D> {
  return up(core.copy(a)) as NDArray<D>;
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

export function diag<D extends DType>(v: NDArrayCore<D>, k: number = 0): NDArray<D> {
  return up(core.diag(v, k)) as NDArray<D>;
}

export function diagflat<D extends DType>(v: NDArrayCore<D>, k: number = 0): NDArray<D> {
  return up(core.diagflat(v, k)) as NDArray<D>;
}

export function tri<D extends DType = 'float64'>(
  N: number,
  M?: number,
  k: number = 0,
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.tri(N, M, k, dtype)) as NDArray<D>;
}

export function tril(m: NDArrayCore, k: number = 0): NDArray {
  return up(core.tril(m, k));
}

export function triu(m: NDArrayCore, k: number = 0): NDArray {
  return up(core.triu(m, k));
}

export function vander<D extends DType>(
  x: NDArrayCore<D>,
  N?: number,
  increasing: boolean = false,
): NDArray<Promote<D, 'int64'>> {
  return up(core.vander(x, N, increasing)) as NDArray<Promote<D, 'int64'>>;
}

export function frombuffer<D extends DType = DType>(
  buffer: ArrayBuffer | TypedArray,
  dtype: D = DEFAULT_DTYPE as D,
  count: number = -1,
  offset: number = 0,
): NDArray<D> {
  return up(core.frombuffer(buffer, dtype, count, offset)) as NDArray<D>;
}

export function fromfunction<D extends DType = 'float64'>(
  func: (...indices: number[]) => number,
  shape: number[],
  dtype: D = DEFAULT_DTYPE as D,
): NDArray<D> {
  return up(core.fromfunction(func, shape, dtype)) as NDArray<D>;
}

export function fromiter<D extends DType = DType>(
  iter: Iterable<number>,
  dtype: D = DEFAULT_DTYPE as D,
  count: number = -1,
): NDArray<D> {
  return up(core.fromiter(iter, dtype, count)) as NDArray<D>;
}

export function fromstring<D extends DType = DType>(
  string: string,
  dtype: D = DEFAULT_DTYPE as D,
  count: number = -1,
  sep?: string,
): NDArray<D> {
  return up(core.fromstring(string, dtype, count, sep)) as NDArray<D>;
}

export function fromfile(
  _file: string,
  _dtype: DType = DEFAULT_DTYPE,
  _count: number = -1,
  _sep: string = '',
): NDArray {
  return up(core.fromfile(_file, _dtype, _count, _sep));
}

/** Calculate n-th discrete difference */
export function diff<D extends DType>(
  a: NDArrayCore<D>,
  n?: number,
  axis?: number,
  prepend?: ArrayLike,
  append?: ArrayLike,
): NDArray<D> {
  return up(core.diff(a, n, axis, prepend, append)) as NDArray<D>;
}

/** Difference between consecutive elements in 1D array */
export function ediff1d<D extends DType>(
  a: NDArrayCore<D>,
  to_end?: number[] | null,
  to_begin?: number[] | null,
): NDArray<D> {
  return up(core.ediff1d(a, to_end, to_begin)) as NDArray<D>;
}

/** Dot product of two arrays */
export function dot<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
): NDArray<Promote<A, B>> | Scalar<Promote<A, B>> {
  const r = core.dot(a, b);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<Promote<A, B>>
    | Scalar<Promote<A, B>>;
}

/** Matrix trace (sum of diagonal elements, supports ND batch) */
export function trace<D extends DType>(
  a: NDArrayCore<D>,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1,
): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
  const r = core.trace(a, offset, axis1, axis2);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<ReductionAccum<D>>
    | Scalar<ReductionAccum<D>>;
}

/** Return diagonal or construct diagonal array */
export function diagonal<D extends DType>(
  a: NDArrayCore<D>,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1,
): NDArray<D> {
  return up(core.diagonal(a, offset, axis1, axis2)) as NDArray<D>;
}

/** Kronecker product */
export function kron<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function kron(a: NDArrayCore, b: NDArrayCore): NDArray {
  return up(core.kron(a, b));
}

/** Transpose array - returns a view */
export function transpose<D extends DType>(a: NDArrayCore<D>, axes?: number[]): NDArray<D> {
  return up(core.transpose(a, axes)) as NDArray<D>;
}

/** Inner product of two arrays */
export function inner<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
): NDArray<Promote<A, B>> | Scalar<Promote<A, B>> {
  const r = core.inner(a, b);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<Promote<A, B>>
    | Scalar<Promote<A, B>>;
}

/** Outer product of two arrays */
export function outer<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function outer(a: NDArrayCore, b: NDArrayCore): NDArray {
  return up(core.outer(a, b));
}

/** Tensor dot product */
export function tensordot<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
  axes: number | [number[], number[]] = 2,
): NDArray<Promote<A, B>> | Scalar<Promote<A, B>> {
  const r = core.tensordot(a, b, axes);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<Promote<A, B>>
    | Scalar<Promote<A, B>>;
}

/** Vector dot product along specified axis */
export function vecdot<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
  axis: number = -1,
): NDArray<Promote<A, B>> | Scalar<Promote<A, B>> {
  const r = core.vecdot(x1, x2, axis);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<Promote<A, B>>
    | Scalar<Promote<A, B>>;
}

/** Matrix transpose (swap last two axes) */
/** Matrix transpose - returns a view */
export function matrix_transpose<D extends DType>(a: NDArrayCore<D>): NDArray<D> {
  return up(core.matrix_transpose(a)) as NDArray<D>;
}

/** Permute array dimensions - returns a view */
export function permute_dims<D extends DType>(a: NDArrayCore<D>, axes?: number[]): NDArray<D> {
  return up(core.permute_dims(a, axes)) as NDArray<D>;
}

/** Matrix-vector product */
export function matvec<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function matvec(x1: NDArrayCore, x2: NDArrayCore): NDArray {
  return up(core.matvec(x1, x2));
}

/** Vector-matrix product */
export function vecmat<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function vecmat(x1: NDArrayCore, x2: NDArrayCore): NDArray {
  return up(core.vecmat(x1, x2));
}

/** Cross product */
export function cross<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
  axisa: number = -1,
  axisb: number = -1,
  axisc: number = -1,
  axis?: number,
): NDArray<Promote<A, B>> {
  return ((): NDArray => {
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
    if (typeof r === 'number' || typeof r === 'bigint') {
      if (isComplexDType(dtype)) {
        const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
        const Ctor = getTypedArrayConstructor(baseDtype)!;
        const data = new Ctor(2);
        data[0] = Number(r);
        data[1] = 0;
        return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
      }
      const Ctor = getTypedArrayConstructor(dtype)!;
      const data = new Ctor(1);
      data[0] = r as never;
      return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
    }
    return up(r);
  })() as unknown as NDArray<Promote<A, B>>;
}

/** Matrix multiplication */
export function matmul<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function matmul(a: NDArrayCore, b: NDArrayCore): NDArray {
  return up(core.matmul(a, b));
}

/** Element-wise logical AND */
export function logical_and<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function logical_and<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function logical_and(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logical_and(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise logical OR */
export function logical_or<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function logical_or<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function logical_or(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logical_or(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise logical NOT */
export function logical_not<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.logical_not(x)) as NDArray<'bool'>;
}

/** Element-wise logical XOR */
export function logical_xor<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function logical_xor<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function logical_xor(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.logical_xor(x1, x2)) as NDArray<'bool'>;
}

/** Test for finite values */
export function isfinite<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isfinite(x)) as NDArray<'bool'>;
}

/** Test for infinity */
export function isinf<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isinf(x)) as NDArray<'bool'>;
}

/** Test for NaN */
export function isnan<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isnan(x)) as NDArray<'bool'>;
}

/** Test for NaT (not a time) - returns all False for numeric arrays */
export function isnat<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isnat(x)) as NDArray<'bool'>;
}

/** Test for negative infinity */
export function isneginf<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isneginf(x)) as NDArray<'bool'>;
}

/** Test for positive infinity */
export function isposinf<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isposinf(x)) as NDArray<'bool'>;
}

/** Element-wise test for complex values */
export function iscomplex<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.iscomplex(x)) as NDArray<'bool'>;
}

/** Element-wise test for real values */
export function isreal<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.isreal(x)) as NDArray<'bool'>;
}

/** Return real array if imaginary part is negligible */
export function real_if_close(x: NDArrayCore, tol?: number): NDArray {
  return up(core.real_if_close(x, tol));
}

/** Return sign of x1 with magnitude of x2 */
export function copysign<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function copysign<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function copysign(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.copysign(x1, x2));
}

/** Test for negative sign bit */
export function signbit<D extends DType>(x: NDArrayCore<D>): NDArray<'bool'> {
  return up(core.signbit(x)) as NDArray<'bool'>;
}

/** Next floating-point value toward x2 */
export function nextafter<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function nextafter<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function nextafter(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.nextafter(x1, x2));
}

/** Spacing between x and nearest adjacent number */
export function spacing<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.spacing(x)) as NDArray<MathResult<D>>;
}

/** Element-wise greater than comparison */
export function greater<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function greater<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function greater(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.greater(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise greater than or equal comparison */
export function greater_equal<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function greater_equal<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function greater_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.greater_equal(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise less than comparison */
export function less<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function less<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function less(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.less(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise less than or equal comparison */
export function less_equal<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function less_equal<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function less_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.less_equal(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise equality comparison */
export function equal<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function equal<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.equal(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise inequality comparison */
export function not_equal<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<'bool'>;
export function not_equal<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<'bool'>;
export function not_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.not_equal(x1, x2)) as NDArray<'bool'>;
}

/** Element-wise close comparison with tolerance */
export function isclose<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  b: NDArrayCore<B>,
  rtol?: number,
  atol?: number,
): NDArray<'bool'>;
export function isclose<A extends DType>(
  a: NDArrayCore<A>,
  b: number,
  rtol?: number,
  atol?: number,
): NDArray<'bool'>;
export function isclose(
  a: NDArrayCore,
  b: NDArrayCore | number,
  rtol: number = 1e-5,
  atol: number = 1e-8,
): NDArray {
  return up(core.isclose(a, b, rtol, atol)) as NDArray<'bool'>;
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
export function polyadd<A extends DType = 'float64', B extends DType = 'float64'>(
  a1: NDArrayCore<A> | number[],
  a2: NDArrayCore<B> | number[],
): NDArray<Promote<A, B>> {
  return up(core.polyadd(a1, a2)) as unknown as NDArray<Promote<A, B>>;
}

/**
 * Differentiate a polynomial
 */
export function polyder<D extends DType = 'float64'>(
  p: NDArrayCore<D> | number[],
  m: number = 1,
): NDArray<Promote<D, 'int64'>> {
  return up(core.polyder(p, m)) as unknown as NDArray<Promote<D, 'int64'>>;
}

/**
 * Divide two polynomials
 */
export function polydiv<A extends DType = 'float64', B extends DType = 'float64'>(
  u: NDArrayCore<A> | number[],
  v: NDArrayCore<B> | number[],
): [NDArray<Divide<A, B>>, NDArray<Divide<A, B>>] {
  const r = core.polydiv(u, v);
  return [up(r[0]), up(r[1])] as unknown as [NDArray<Divide<A, B>>, NDArray<Divide<A, B>>];
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
export function polyint<D extends DType = 'float64'>(
  p: NDArrayCore<D> | number[],
  m: number = 1,
  k: number | number[] = 0,
): NDArray<Promote<D, 'float64'>> {
  return up(core.polyint(p, m, k)) as unknown as NDArray<Promote<D, 'float64'>>;
}

/**
 * Multiply two polynomials
 */
export function polymul<A extends DType = 'float64', B extends DType = 'float64'>(
  a1: NDArrayCore<A> | number[],
  a2: NDArrayCore<B> | number[],
): NDArray<Promote<A, B>> {
  return up(core.polymul(a1, a2)) as unknown as NDArray<Promote<A, B>>;
}

/**
 * Subtract two polynomials
 */
export function polysub<A extends DType = 'float64', B extends DType = 'float64'>(
  a1: NDArrayCore<A> | number[],
  a2: NDArrayCore<B> | number[],
): NDArray<Promote<A, B>> {
  return up(core.polysub(a1, a2)) as unknown as NDArray<Promote<A, B>>;
}

/**
 * Evaluate a polynomial at given points
 */
export function polyval(
  p: NDArrayCore | number[],
  x: NDArrayCore | number | number[],
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
export function sum<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
  const r = core.sum(a, axis, keepdims, opts);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<ReductionAccum<D>>
    | Scalar<ReductionAccum<D>>;
}

/** Mean of array elements */
export function mean<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.mean(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Product of array elements */
export function prod<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
  const r = core.prod(a, axis, keepdims, opts);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<ReductionAccum<D>>
    | Scalar<ReductionAccum<D>>;
}

/** Maximum of array elements */
export function max<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArray<D> | Scalar<D> {
  const r = core.max(a, axis, keepdims, opts);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
}

/** Minimum of array elements */
export function min<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArray<D> | Scalar<D> {
  const r = core.min(a, axis, keepdims, opts);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
}

/** Peak-to-peak (max - min) */
export function ptp<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<D> | Scalar<D> {
  const r = core.ptp(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
}

/** Index of minimum value */
export function argmin<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
  keepdims?: boolean,
): NDArray<'int32'> | number {
  const r = core.argmin(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int32'> | number;
}

/** Index of maximum value */
export function argmax<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
  keepdims?: boolean,
): NDArray<'int32'> | number {
  const r = core.argmax(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int32'> | number;
}

/** Variance of array elements */
export function variance<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean,
): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
  const r = core.variance(a, axis, ddof, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<StdVar<D>>
    | Scalar<StdVar<D>>;
}

/** Standard deviation of array elements */
export function std<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean,
): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
  const r = core.std(a, axis, ddof, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<StdVar<D>>
    | Scalar<StdVar<D>>;
}

/** Median of array elements */
export function median<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.median(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Percentile of array elements */
export function percentile<D extends DType>(
  a: NDArrayCore<D>,
  q: number,
  axis?: number,
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.percentile(a, q, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Quantile of array elements */
export function quantile<D extends DType>(
  a: NDArrayCore<D>,
  q: number,
  axis?: number,
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.quantile(a, q, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/**
 * Weighted average. When `returned` is true, returns `[avg, sum_of_weights]`
 * (matching `np.average(..., returned=True)`); otherwise returns just `avg`.
 */
export function average<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
  weights?: NDArrayCore,
  keepdims?: boolean,
  returned?: boolean,
):
  | NDArray<TrueDivide<D>>
  | Scalar<TrueDivide<D>>
  | [
      NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>,
      NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>,
    ] {
  const r = core.average(a, axis, weights, keepdims, returned);
  if (Array.isArray(r)) {
    const [avg, sw] = r;
    const upAvg = avg instanceof NDArrayCore ? up(avg) : avg;
    const upSw = sw instanceof NDArrayCore ? up(sw) : sw;
    return [upAvg, upSw] as unknown as [
      NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>,
      NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>,
    ];
  }
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Test if all elements are true */
export function all<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArray<'bool'> | boolean {
  const r = core.all(a, axis, keepdims, opts);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'bool'> | boolean;
}

/** Test if any element is true */
export function any<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArray<'bool'> | boolean {
  const r = core.any(a, axis, keepdims, opts);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'bool'> | boolean;
}

/** Cumulative sum */
export function cumsum<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
): NDArray<ReductionAccum<D>> {
  return up(core.cumsum(a, axis)) as NDArray<ReductionAccum<D>>;
}

/** Cumulative product */
export function cumprod<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
): NDArray<ReductionAccum<D>> {
  return up(core.cumprod(a, axis)) as NDArray<ReductionAccum<D>>;
}

/** Sum ignoring NaN */
export function nansum<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
  const r = core.nansum(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<ReductionAccum<D>>
    | Scalar<ReductionAccum<D>>;
}

/** Product ignoring NaN */
export function nanprod<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
  const r = core.nanprod(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<ReductionAccum<D>>
    | Scalar<ReductionAccum<D>>;
}

/** Mean ignoring NaN */
export function nanmean<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.nanmean(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Variance ignoring NaN */
export function nanvar<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean,
): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
  const r = core.nanvar(a, axis, ddof, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<StdVar<D>>
    | Scalar<StdVar<D>>;
}

/** Standard deviation ignoring NaN */
export function nanstd<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean,
): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
  const r = core.nanstd(a, axis, ddof, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<StdVar<D>>
    | Scalar<StdVar<D>>;
}

/** Minimum ignoring NaN */
export function nanmin<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<D> | Scalar<D> {
  const r = core.nanmin(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
}

/** Maximum ignoring NaN */
export function nanmax<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number | number[],
  keepdims?: boolean,
): NDArray<D> | Scalar<D> {
  const r = core.nanmax(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
}

/** Index of minimum ignoring NaN */
export function nanargmin<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
): NDArray<'int32'> | number {
  const r = core.nanargmin(a, axis);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int32'> | number;
}

/** Index of maximum ignoring NaN */
export function nanargmax<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
): NDArray<'int32'> | number {
  const r = core.nanargmax(a, axis);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int32'> | number;
}

/** Cumulative sum ignoring NaN */
export function nancumsum<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
): NDArray<ReductionAccum<D>> {
  return up(core.nancumsum(a, axis)) as NDArray<ReductionAccum<D>>;
}

/** Cumulative product ignoring NaN */
export function nancumprod<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
): NDArray<ReductionAccum<D>> {
  return up(core.nancumprod(a, axis)) as NDArray<ReductionAccum<D>>;
}

/** Median ignoring NaN */
export function nanmedian<D extends DType>(
  a: NDArrayCore<D>,
  axis?: number,
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.nanmedian(a, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Quantile ignoring NaN */
export function nanquantile<D extends DType>(
  a: NDArrayCore<D>,
  q: number,
  axis?: number,
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.nanquantile(a, q, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Percentile ignoring NaN */
export function nanpercentile<D extends DType>(
  a: NDArrayCore<D>,
  q: number,
  axis?: number,
  keepdims?: boolean,
): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
  const r = core.nanpercentile(a, q, axis, keepdims);
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;
}

/** Round to given decimals */
export function around<D extends DType>(
  a: NDArrayCore<D>,
  decimals: number = 0,
): NDArray<RoundResult<D>> {
  return up(core.around(a, decimals)) as NDArray<RoundResult<D>>;
}

/** Round (same as around) */
export function round<D extends DType>(
  a: NDArrayCore<D>,
  decimals: number = 0,
): NDArray<RoundResult<D>> {
  return up(core.round(a, decimals)) as NDArray<RoundResult<D>>;
}

/** Ceiling */
export function ceil<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.ceil(x)) as NDArray<MathResult<D>>;
}

/** Round toward zero */
export function fix<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.fix(x)) as NDArray<MathResult<D>>;
}

/** Floor */
export function floor<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.floor(x)) as NDArray<MathResult<D>>;
}

/** Round to nearest integer */
export function rint<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.rint(x)) as NDArray<MathResult<D>>;
}

/** Truncate toward zero */
export function trunc<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.trunc(x)) as NDArray<MathResult<D>>;
}

/**
 * Test if elements in ar1 are in ar2.
 * @deprecated Use {@link isin} instead. `in1d` follows NumPy's deprecation and will be removed in a future release.
 */
export function in1d<A extends DType, B extends DType>(
  ar1: NDArrayCore<A>,
  ar2: NDArrayCore<B>,
): NDArray<'bool'>;
export function in1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.in1d(ar1, ar2)) as NDArray<'bool'>;
}

/** Find intersection of two arrays */
export function intersect1d<A extends DType, B extends DType>(
  ar1: NDArrayCore<A>,
  ar2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function intersect1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.intersect1d(ar1, ar2));
}

/** Test if elements are in test elements */
export function isin<A extends DType, B extends DType>(
  element: NDArrayCore<A>,
  testElements: NDArrayCore<B>,
): NDArray<'bool'>;
export function isin(element: NDArrayCore, testElements: NDArrayCore): NDArray {
  return up(core.isin(element, testElements)) as NDArray<'bool'>;
}

/** Find set difference (elements in ar1 not in ar2) */
export function setdiff1d<D extends DType>(ar1: NDArrayCore<D>, ar2: NDArrayCore): NDArray<D> {
  return up(core.setdiff1d(ar1, ar2)) as NDArray<D>;
}

/** Find symmetric difference (elements in either but not both) */
export function setxor1d<A extends DType, B extends DType>(
  ar1: NDArrayCore<A>,
  ar2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function setxor1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.setxor1d(ar1, ar2));
}

/** Find union of two arrays */
export function union1d<A extends DType, B extends DType>(
  ar1: NDArrayCore<A>,
  ar2: NDArrayCore<B>,
): NDArray<Promote<A, B>>;
export function union1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArray {
  return up(core.union1d(ar1, ar2));
}

/** Trim leading/trailing zeros */
export function trim_zeros<D extends DType>(
  filt: NDArrayCore<D>,
  trim?: 'f' | 'b' | 'fb',
): NDArray<D> {
  return up(core.trim_zeros(filt, trim)) as NDArray<D>;
}

/** Return unique values (Array API) */
export function unique_values<D extends DType>(x: NDArrayCore<D>): NDArray<D> {
  return up(core.unique_values(x)) as NDArray<D>;
}

/**
 * Append values to the end of an array
 */
export function append(
  arr: NDArrayCore,
  values: NDArrayCore | number | number[],
  axis?: number,
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
  axis?: number,
): NDArray {
  return up(core.insert(arr, obj, values, axis));
}

/**
 * Pad an array
 */
export function pad<D extends DType>(
  arr: NDArrayCore<D>,
  pad_width: PadWidthArg,
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
  constant_values: PadValueArg = 0,
): NDArray<D> {
  return up(core.pad(arr, pad_width, mode, constant_values)) as NDArray<D>;
}

/** Reshape array without changing data */
export function reshape<D extends DType>(a: NDArrayCore<D>, newShape: number[]): NDArray<D> {
  return up(core.reshape(a, newShape)) as NDArray<D>;
}

/** Flatten array to 1D */
export function flatten<D extends DType>(a: NDArrayCore<D>): NDArray<D> {
  return up(core.flatten(a)) as NDArray<D>;
}

/** Return contiguous flattened array */
export function ravel<D extends DType>(a: NDArrayCore<D>): NDArray<D> {
  return up(core.ravel(a)) as NDArray<D>;
}

/** Remove single-dimensional entries from shape - returns a view */
export function squeeze<D extends DType>(a: NDArrayCore<D>, axis?: number): NDArray<D> {
  return up(core.squeeze(a, axis)) as NDArray<D>;
}

/** Expand array dimensions - returns a view */
export function expand_dims<D extends DType>(a: NDArrayCore<D>, axis: number): NDArray<D> {
  return up(core.expand_dims(a, axis)) as NDArray<D>;
}

/** Interchange two axes - returns a view */
export function swapaxes<D extends DType>(
  a: NDArrayCore<D>,
  axis1: number,
  axis2: number,
): NDArray<D> {
  return up(core.swapaxes(a, axis1, axis2)) as NDArray<D>;
}

/** Move axis to new position - returns a view */
export function moveaxis<D extends DType>(
  a: NDArrayCore<D>,
  source: number | number[],
  destination: number | number[],
): NDArray<D> {
  return up(core.moveaxis(a, source, destination)) as NDArray<D>;
}

/** Roll axis to given position - returns a view */
export function rollaxis<D extends DType>(
  a: NDArrayCore<D>,
  axis: number,
  start: number = 0,
): NDArray<D> {
  return up(core.rollaxis(a, axis, start)) as NDArray<D>;
}

/**
 * Join arrays along an existing axis. Pass `axis=null` to flatten each input
 * and concatenate the result along axis 0 (matches `np.concatenate(..., axis=None)`).
 */
export function concatenate(arrays: NDArrayCore[], axis: number | null = 0): NDArray {
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
export function concat(arrays: NDArrayCore[], axis: number | null = 0): NDArray {
  return up(core.concat(arrays, axis));
}

/** Stack 1D arrays as columns */
export function column_stack(arrays: NDArrayCore[]): NDArray {
  return up(core.column_stack(arrays));
}

/** Assemble arrays from nested sequences of blocks (np.block semantics) */
export function block(arrays: NestedNDArrays[]): NDArray {
  return up(core.block(arrays));
}

/** Unstack array into list of arrays */
export function unstack<D extends DType>(a: NDArrayCore<D>, axis: number = 0): NDArray<D>[] {
  return core.unstack(a, axis).map(up) as NDArray<D>[];
}

/** Tile array by given repetitions */
export function tile<D extends DType>(a: NDArrayCore<D>, reps: number | number[]): NDArray<D> {
  return up(core.tile(a, reps)) as NDArray<D>;
}

/** Repeat elements of array */
export function repeat<D extends DType>(
  a: NDArrayCore<D>,
  repeats: number | number[],
  axis?: number,
): NDArray<D> {
  return up(core.repeat(a, repeats, axis)) as NDArray<D>;
}

/** Reverse order of elements along axis */
export function flip<D extends DType>(m: NDArrayCore<D>, axis?: number | number[]): NDArray<D> {
  return up(core.flip(m, axis)) as NDArray<D>;
}

/** Flip array left/right */
export function fliplr<D extends DType>(m: NDArrayCore<D>): NDArray<D> {
  return up(core.fliplr(m)) as NDArray<D>;
}

/** Flip array up/down */
export function flipud<D extends DType>(m: NDArrayCore<D>): NDArray<D> {
  return up(core.flipud(m)) as NDArray<D>;
}

/** Rotate array by 90 degrees */
export function rot90<D extends DType>(
  m: NDArrayCore<D>,
  k: number = 1,
  axes: [number, number] = [0, 1],
): NDArray<D> {
  return up(core.rot90(m, k, axes)) as NDArray<D>;
}

/** Roll array elements along axis */
export function roll<D extends DType>(
  a: NDArrayCore<D>,
  shift: number | number[],
  axis?: number | number[],
): NDArray<D> {
  return up(core.roll(a, shift, axis)) as NDArray<D>;
}

/** Resize array to new shape */
export function resize<D extends DType>(a: NDArrayCore<D>, new_shape: number[]): NDArray<D> {
  return up(core.resize(a, new_shape)) as NDArray<D>;
}

/** Sort array along axis */
export function sort(a: NDArrayCore, axis: number = -1): NDArray {
  return up(core.sort(a, axis));
}

/** Indices that would sort array */
export function argsort<D extends DType>(a: NDArrayCore<D>, axis: number = -1): NDArray<'float64'> {
  return up(core.argsort(a, axis)) as NDArray<'float64'>;
}

/** Indirect stable sort on multiple keys */
export function lexsort<D extends DType>(keys: NDArrayCore<D>[]): NDArray<'float64'> {
  return up(core.lexsort(keys)) as NDArray<'float64'>;
}

/** Partially sort array */
export function partition(a: NDArrayCore, kth: number, axis: number = -1): NDArray {
  return up(core.partition(a, kth, axis));
}

/** Indices that would partially sort array */
export function argpartition<D extends DType>(
  a: NDArrayCore<D>,
  kth: number,
  axis: number = -1,
): NDArray<'float64'> {
  return up(core.argpartition(a, kth, axis)) as NDArray<'float64'>;
}

/** Sort complex array */
export function sort_complex(a: NDArrayCore): NDArray {
  return up(core.sort_complex(a));
}

/** Indices of non-zero elements */
export function nonzero(a: NDArrayCore): NDArray<'float64'>[] {
  return core.nonzero(a).map(up) as NDArray<'float64'>[];
}

/** Indices where condition is True */
export function argwhere<D extends DType>(a: NDArrayCore<D>): NDArray<'float64'> {
  return up(core.argwhere(a)) as NDArray<'float64'>;
}

/** Indices of non-zero elements in flattened array */
export function flatnonzero<D extends DType>(a: NDArrayCore<D>): NDArray<'float64'> {
  return up(core.flatnonzero(a)) as NDArray<'float64'>;
}

/** Extract elements where condition is True */
export function extract(condition: NDArrayCore, a: NDArrayCore): NDArray {
  return up(core.extract(condition, a));
}

/** Count occurrences of values */
export function bincount(x: ArrayLike, weights?: ArrayLike, minlength?: number): NDArray {
  return up(core.bincount(x, weights, minlength));
}

/** Digitize values into bins */
export function digitize(x: ArrayLike, bins: ArrayLike, right?: boolean): NDArray {
  return up(core.digitize(x, bins, right));
}

/** Cross-correlation */
export function correlate<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  v: NDArrayCore<B>,
  mode?: 'valid' | 'same' | 'full',
): NDArray<Promote<A, B>>;
export function correlate(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full',
): NDArray {
  return up(core.correlate(a, v, mode));
}

/** Convolution */
export function convolve<A extends DType, B extends DType>(
  a: NDArrayCore<A>,
  v: NDArrayCore<B>,
  mode?: 'valid' | 'same' | 'full',
): NDArray<Promote<A, B>>;
export function convolve(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full',
): NDArray {
  return up(core.convolve(a, v, mode));
}

/** Covariance matrix */
export function cov<D extends DType>(
  m: NDArrayCore<D>,
  y?: NDArrayCore,
  rowvar?: boolean,
  bias?: boolean,
  ddof?: number,
): NDArray<Promote<D, 'float64'>> {
  return up(core.cov(m, y, rowvar, bias, ddof)) as NDArray<Promote<D, 'float64'>>;
}

/** Correlation coefficients */
export function corrcoef<D extends DType>(
  x: NDArrayCore<D>,
  y?: NDArrayCore,
  rowvar?: boolean,
): NDArray<Promote<D, 'float64'>> {
  return up(core.corrcoef(x, y, rowvar)) as NDArray<Promote<D, 'float64'>>;
}

/** Integrate using trapezoidal rule */
export function trapezoid(
  y: NDArrayCore,
  x?: NDArrayCore,
  dx?: number,
  axis?: number,
): NDArray | number | Complex {
  const r = core.trapezoid(y, x, dx, axis);
  return r instanceof NDArrayCore ? up(r) : r;
}

/** Sine of array elements */
export function sin<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.sin(x)) as NDArray<MathResult<D>>;
}

/** Cosine of array elements */
export function cos<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.cos(x)) as NDArray<MathResult<D>>;
}

/** Tangent of array elements */
export function tan<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.tan(x)) as NDArray<MathResult<D>>;
}

/** Inverse sine */
export function arcsin<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.arcsin(x)) as NDArray<MathResult<D>>;
}

/** Inverse cosine */
export function arccos<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.arccos(x)) as NDArray<MathResult<D>>;
}

/** Inverse tangent */
export function arctan<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.arctan(x)) as NDArray<MathResult<D>>;
}

/** Two-argument inverse tangent */
export function arctan2<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function arctan2<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function arctan2(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.arctan2(x1, x2));
}

/** Hypotenuse (sqrt(x1^2 + x2^2)) */
export function hypot<A extends DType, B extends DType>(
  x1: NDArrayCore<A>,
  x2: NDArrayCore<B>,
): NDArray<MathBinary<A, B>>;
export function hypot<A extends DType>(x1: NDArrayCore<A>, x2: number): NDArray<MathResult<A>>;
export function hypot(x1: NDArrayCore, x2: NDArrayCore | number): NDArray {
  return up(core.hypot(x1, x2));
}

/** Convert radians to degrees */
export function degrees<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.degrees(x)) as NDArray<MathResult<D>>;
}

/** Convert degrees to radians */
export function radians<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.radians(x)) as NDArray<MathResult<D>>;
}

/** Convert degrees to radians (alias) */
export function deg2rad<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.deg2rad(x)) as NDArray<MathResult<D>>;
}

/** Convert radians to degrees (alias) */
export function rad2deg<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.rad2deg(x)) as NDArray<MathResult<D>>;
}

/** Hyperbolic sine */
export function sinh<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.sinh(x)) as NDArray<MathResult<D>>;
}

/** Hyperbolic cosine */
export function cosh<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.cosh(x)) as NDArray<MathResult<D>>;
}

/** Hyperbolic tangent */
export function tanh<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.tanh(x)) as NDArray<MathResult<D>>;
}

/** Inverse hyperbolic sine */
export function arcsinh<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.arcsinh(x)) as NDArray<MathResult<D>>;
}

/** Inverse hyperbolic cosine */
export function arccosh<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.arccosh(x)) as NDArray<MathResult<D>>;
}

/** Inverse hyperbolic tangent */
export function arctanh<D extends DType>(x: NDArrayCore<D>): NDArray<MathResult<D>> {
  return up(core.arctanh(x)) as NDArray<MathResult<D>>;
}

// ============================================================
// Re-exported Functions (no wrapping needed)
// ============================================================

// Runtime capabilities
export { hasFloat16 } from '../common/dtype';

// WASM configuration
export { wasmConfig } from '../common/wasm/config';
export { configureWasm, wasmFreeBytes } from '../common/wasm/runtime';
export {
  abs,
  acos,
  acosh,
  allclose,
  amax,
  amin,
  array_equal,
  array_equiv,
  array_repr,
  array_split,
  array_str,
  array2string,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  base_repr,
  binary_repr,
  broadcast_arrays,
  broadcast_shapes,
  byteswap,
  can_cast,
  common_type,
  conjugate,
  copyto,
  count_nonzero,
  cumulative_prod,
  cumulative_sum,
  dsplit,
  einsum,
  einsum_path,
  fill,
  fill_diagonal,
  format_float_positional,
  format_float_scientific,
  get_printoptions,
  geterr,
  gradient,
  histogram,
  histogram_bin_edges,
  histogram2d,
  histogramdd,
  hsplit,
  iscomplexobj,
  isdtype,
  isfortran,
  isrealobj,
  isscalar,
  issubdtype,
  item,
  iterable,
  ix_,
  linalg,
  may_share_memory,
  min_scalar_type,
  mintypecode,
  ndim,
  place,
  pow,
  printoptions,
  promote_types,
  put,
  put_along_axis,
  putmask,
  result_type,
  row_stack,
  searchsorted,
  set_printoptions,
  seterr,
  shape,
  shares_memory,
  size,
  split,
  tobytes,
  tofile,
  tolist,
  true_divide,
  typename,
  unique,
  unique_all,
  unique_counts,
  unique_inverse,
  var_,
  vdot,
  view,
  vsplit,
  where,
} from '../core';
