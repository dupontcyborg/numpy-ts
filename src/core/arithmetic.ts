/**
 * Arithmetic Functions - Tree-shakeable standalone functions
 *
 * This module provides arithmetic and mathematical functions that can be
 * imported independently for optimal tree-shaking.
 *
 * Import from here for minimal bundle size:
 *   import { sqrt, exp, abs } from 'numpy-ts/functions/arithmetic';
 *
 * Or import from main entry point (also tree-shakeable):
 *   import { sqrt, exp, abs } from 'numpy-ts';
 */

import * as arithmeticOps from '../common/ops/arithmetic';
import * as exponentialOps from '../common/ops/exponential';
import { NDArrayCore, toStorage, fromStorage, fromStorageTuple } from './types';

// ============================================================
// Basic Arithmetic Functions
// ============================================================

/** Add arguments element-wise */
export function add(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.add(toStorage(x1), x2Storage));
}

/** Subtract arguments element-wise */
export function subtract(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.subtract(toStorage(x1), x2Storage));
}

/** Multiply arguments element-wise */
export function multiply(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.multiply(toStorage(x1), x2Storage));
}

// ============================================================
// Exponential and Logarithmic Functions
// ============================================================

/** Square root of array elements */
export function sqrt(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.sqrt(toStorage(x)));
}

/** Element-wise power */
export function power(x: NDArrayCore, exponent: NDArrayCore | number): NDArrayCore {
  const expStorage = typeof exponent === 'number' ? exponent : toStorage(exponent);
  return fromStorage(exponentialOps.power(toStorage(x), expStorage));
}

/** Alias for power */
export const pow = power;

/** Exponential (e^x) */
export function exp(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.exp(toStorage(x)));
}

/** 2^x */
export function exp2(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.exp2(toStorage(x)));
}

/** exp(x) - 1 */
export function expm1(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.expm1(toStorage(x)));
}

/** Natural logarithm */
export function log(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.log(toStorage(x)));
}

/** Base-2 logarithm */
export function log2(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.log2(toStorage(x)));
}

/** Base-10 logarithm */
export function log10(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.log10(toStorage(x)));
}

/** log(1 + x) */
export function log1p(x: NDArrayCore): NDArrayCore {
  return fromStorage(exponentialOps.log1p(toStorage(x)));
}

/** log(exp(x1) + exp(x2)) */
export function logaddexp(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(exponentialOps.logaddexp(toStorage(x1), x2Storage));
}

/** log2(2^x1 + 2^x2) */
export function logaddexp2(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(exponentialOps.logaddexp2(toStorage(x1), x2Storage));
}

// ============================================================
// Basic Arithmetic Functions
// ============================================================

/** Absolute value */
export function absolute(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.absolute(toStorage(x)));
}

/** Alias for absolute */
export const abs = absolute;

/** Numerical negative */
export function negative(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.negative(toStorage(x)));
}

/** Element-wise sign */
export function sign(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.sign(toStorage(x)));
}

/** Element-wise modulo */
export function mod(x: NDArrayCore, divisor: NDArrayCore | number): NDArrayCore {
  const divStorage = typeof divisor === 'number' ? divisor : toStorage(divisor);
  return fromStorage(arithmeticOps.mod(toStorage(x), divStorage));
}

/** Element-wise division */
export function divide(x: NDArrayCore, divisor: NDArrayCore | number): NDArrayCore {
  const divStorage = typeof divisor === 'number' ? divisor : toStorage(divisor);
  return fromStorage(arithmeticOps.divide(toStorage(x), divStorage));
}

/** Alias for divide */
export const true_divide = divide;

/** Element-wise floor division */
export function floor_divide(x: NDArrayCore, divisor: NDArrayCore | number): NDArrayCore {
  const divStorage = typeof divisor === 'number' ? divisor : toStorage(divisor);
  return fromStorage(arithmeticOps.floorDivide(toStorage(x), divStorage));
}

/** Numerical positive (returns copy) */
export function positive(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.positive(toStorage(x)));
}

/** Reciprocal (1/x) */
export function reciprocal(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.reciprocal(toStorage(x)));
}

/** Cube root */
export function cbrt(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.cbrt(toStorage(x)));
}

/** Absolute value (float) */
export function fabs(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.fabs(toStorage(x)));
}

/** Element-wise divmod (quotient and remainder) */
export function divmod(x: NDArrayCore, y: NDArrayCore | number): [NDArrayCore, NDArrayCore] {
  const yStorage = typeof y === 'number' ? y : toStorage(y);
  return fromStorageTuple(arithmeticOps.divmod(toStorage(x), yStorage));
}

/** Element-wise square */
export function square(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.square(toStorage(x)));
}

/** Element-wise remainder */
export function remainder(x: NDArrayCore, y: NDArrayCore | number): NDArrayCore {
  const yStorage = typeof y === 'number' ? y : toStorage(y);
  return fromStorage(arithmeticOps.remainder(toStorage(x), yStorage));
}

/** Heaviside step function */
export function heaviside(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.heaviside(toStorage(x1), x2Storage));
}

/** Float power (always returns float) */
export function float_power(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.float_power(toStorage(x1), x2Storage));
}

/** C-style fmod */
export function fmod(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.fmod(toStorage(x1), x2Storage));
}

/** Decompose into mantissa and exponent */
export function frexp(x: NDArrayCore): [NDArrayCore, NDArrayCore] {
  return fromStorageTuple(arithmeticOps.frexp(toStorage(x)));
}

/** Greatest common divisor */
export function gcd(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.gcd(toStorage(x1), x2Storage));
}

/** Least common multiple */
export function lcm(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.lcm(toStorage(x1), x2Storage));
}

/** Compose from mantissa and exponent */
export function ldexp(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.ldexp(toStorage(x1), x2Storage));
}

/** Decompose into fractional and integer parts */
export function modf(x: NDArrayCore): [NDArrayCore, NDArrayCore] {
  return fromStorageTuple(arithmeticOps.modf(toStorage(x)));
}

// ============================================================
// Comparison and Clipping Functions
// ============================================================

/** Clip values to a range */
export function clip(
  a: NDArrayCore,
  a_min: NDArrayCore | number | null,
  a_max: NDArrayCore | number | null
): NDArrayCore {
  const minStorage = a_min === null ? null : typeof a_min === 'number' ? a_min : toStorage(a_min);
  const maxStorage = a_max === null ? null : typeof a_max === 'number' ? a_max : toStorage(a_max);
  return fromStorage(arithmeticOps.clip(toStorage(a), minStorage, maxStorage));
}

/** Element-wise maximum */
export function maximum(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.maximum(toStorage(x1), x2Storage));
}

/** Element-wise minimum */
export function minimum(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.minimum(toStorage(x1), x2Storage));
}

/** Element-wise maximum (ignores NaN) */
export function fmax(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.fmax(toStorage(x1), x2Storage));
}

/** Element-wise minimum (ignores NaN) */
export function fmin(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(arithmeticOps.fmin(toStorage(x1), x2Storage));
}

/** Replace NaN/Inf with numbers */
export function nan_to_num(
  x: NDArrayCore,
  nan: number = 0.0,
  posinf?: number,
  neginf?: number
): NDArrayCore {
  return fromStorage(arithmeticOps.nan_to_num(toStorage(x), nan, posinf, neginf));
}

/** 1D linear interpolation */
export function interp(
  x: NDArrayCore,
  xp: NDArrayCore,
  fp: NDArrayCore,
  left?: number,
  right?: number
): NDArrayCore {
  return fromStorage(arithmeticOps.interp(toStorage(x), toStorage(xp), toStorage(fp), left, right));
}

/** Unwrap phase angles */
export function unwrap(
  p: NDArrayCore,
  discont: number = Math.PI,
  axis: number = -1,
  period: number = 2 * Math.PI
): NDArrayCore {
  return fromStorage(arithmeticOps.unwrap(toStorage(p), discont, axis, period));
}

/** Sinc function */
export function sinc(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.sinc(toStorage(x)));
}

/** Modified Bessel function of the first kind, order 0 */
export function i0(x: NDArrayCore): NDArrayCore {
  return fromStorage(arithmeticOps.i0(toStorage(x)));
}
