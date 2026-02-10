/**
 * Reduction Functions - Tree-shakeable standalone functions
 *
 * This module provides array reduction and aggregation functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as reductionOps from '../common/ops/reduction';
import { NDArrayCore, toStorage, fromStorage, Complex } from './types';

// ============================================================
// Basic Reductions
// ============================================================

/** Sum of array elements */
export function sum(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | bigint | Complex {
  const result = reductionOps.sum(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Mean of array elements */
export function mean(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.mean(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Product of array elements */
export function prod(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | bigint | Complex {
  const result = reductionOps.prod(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Maximum of array elements */
export function max(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.max(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Alias for max */
export const amax = max;

/** Minimum of array elements */
export function min(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.min(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Alias for min */
export const amin = min;

/** Peak-to-peak (max - min) */
export function ptp(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.ptp(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

// ============================================================
// Index of Extrema
// ============================================================

/** Index of minimum value */
export function argmin(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = reductionOps.argmin(toStorage(a), axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Index of maximum value */
export function argmax(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = reductionOps.argmax(toStorage(a), axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

// ============================================================
// Statistical Reductions
// ============================================================

/** Variance of array elements */
export function variance(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.variance(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Alias for variance */
export const var_ = variance;

/** Standard deviation of array elements */
export function std(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.std(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Median of array elements */
export function median(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | number {
  const result = reductionOps.median(toStorage(a), axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Percentile of array elements */
export function percentile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.percentile(toStorage(a), q, axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Quantile of array elements */
export function quantile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.quantile(toStorage(a), q, axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Weighted average */
export function average(
  a: NDArrayCore,
  axis?: number,
  weights?: NDArrayCore,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const weightsStorage = weights ? toStorage(weights) : undefined;
  const result = reductionOps.average(toStorage(a), axis, weightsStorage, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

// ============================================================
// Logical Reductions
// ============================================================

/** Test if all elements are true */
export function all(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | boolean {
  const result = reductionOps.all(toStorage(a), axis, keepdims);
  if (typeof result === 'boolean') return result;
  return fromStorage(result);
}

/** Test if any element is true */
export function any(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | boolean {
  const result = reductionOps.any(toStorage(a), axis, keepdims);
  if (typeof result === 'boolean') return result;
  return fromStorage(result);
}

// ============================================================
// Cumulative Functions
// ============================================================

/** Cumulative sum */
export function cumsum(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(reductionOps.cumsum(toStorage(a), axis));
}

/** Alias for cumsum */
export const cumulative_sum = cumsum;

/** Cumulative product */
export function cumprod(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(reductionOps.cumprod(toStorage(a), axis));
}

/** Alias for cumprod */
export const cumulative_prod = cumprod;

// ============================================================
// NaN-aware Reductions
// ============================================================

/** Sum ignoring NaN */
export function nansum(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.nansum(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Product ignoring NaN */
export function nanprod(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.nanprod(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Mean ignoring NaN */
export function nanmean(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.nanmean(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Variance ignoring NaN */
export function nanvar(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.nanvar(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Standard deviation ignoring NaN */
export function nanstd(
  a: NDArrayCore,
  axis?: number,
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.nanstd(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Minimum ignoring NaN */
export function nanmin(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.nanmin(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Maximum ignoring NaN */
export function nanmax(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number | Complex {
  const result = reductionOps.nanmax(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Index of minimum ignoring NaN */
export function nanargmin(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = reductionOps.nanargmin(toStorage(a), axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Index of maximum ignoring NaN */
export function nanargmax(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = reductionOps.nanargmax(toStorage(a), axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Cumulative sum ignoring NaN */
export function nancumsum(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(reductionOps.nancumsum(toStorage(a), axis));
}

/** Cumulative product ignoring NaN */
export function nancumprod(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(reductionOps.nancumprod(toStorage(a), axis));
}

/** Median ignoring NaN */
export function nanmedian(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | number {
  const result = reductionOps.nanmedian(toStorage(a), axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Quantile ignoring NaN */
export function nanquantile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.nanquantile(toStorage(a), q, axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Percentile ignoring NaN */
export function nanpercentile(
  a: NDArrayCore,
  q: number,
  axis?: number,
  keepdims?: boolean
): NDArrayCore | number {
  const result = reductionOps.nanpercentile(toStorage(a), q, axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}
