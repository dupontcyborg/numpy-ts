/**
 * Reduction Functions - Tree-shakeable standalone functions
 *
 * This module provides array reduction and aggregation functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as reductionOps from '../common/ops/reduction';
import * as shapeOps from '../common/ops/shape';
import { NDArrayCore, toStorage, fromStorage, Complex, ArrayStorage } from './types';

// ============================================================
// Multi-axis reduction helper
// ============================================================

/**
 * Reduce along multiple axes by repeatedly calling a single-axis reducer.
 * Sorts axes descending so index shifting doesn't affect subsequent reductions.
 * If keepdims=true, re-expands dimensions after reduction.
 */
function reduceMultiAxis<T>(
  a: NDArrayCore,
  axes: number[],
  keepdims: boolean,
  reduceFn: (storage: ArrayStorage, axis: number, keepdims: boolean) => ArrayStorage | T
): NDArrayCore | T {
  const ndim = toStorage(a).ndim;
  // Normalize negative axes
  const normAxes = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  // Sort descending so removing an axis doesn't shift lower-indexed axes
  const sortedDesc = [...normAxes].sort((x, y) => y - x);
  const sortedAsc = [...normAxes].sort((x, y) => x - y);

  let current: ArrayStorage = toStorage(a);
  for (const ax of sortedDesc) {
    const r = reduceFn(current, ax, false);
    if (!(r instanceof ArrayStorage)) {
      if (keepdims) {
        // All dims reduced to scalar — fill a shape-of-all-ones array
        const onesShape = Array(ndim).fill(1) as number[];
        const out = ArrayStorage.zeros(onesShape, current.dtype);
        out.iset(0, r as number | bigint | Complex);
        return fromStorage(out);
      }
      return r as T;
    }
    current = r;
  }

  if (keepdims) {
    // Re-insert size-1 dims at the original axis positions (ascending)
    for (const ax of sortedAsc) {
      current = shapeOps.expandDims(current, ax);
    }
  }

  return fromStorage(current);
}

// ============================================================
// Basic Reductions
// ============================================================

/** Sum of array elements */
export function sum(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | bigint | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) => reductionOps.sum(s, ax, kd));
  }
  const result = reductionOps.sum(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Mean of array elements */
export function mean(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.mean(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
  const result = reductionOps.mean(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Product of array elements */
export function prod(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | bigint | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) => reductionOps.prod(s, ax, kd));
  }
  const result = reductionOps.prod(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Maximum of array elements */
export function max(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.max(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
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
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.min(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
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
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.variance(s, ax, ddof, kd)
    ) as NDArrayCore | number;
  }
  const result = reductionOps.variance(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Alias for variance */
export const var_ = variance;

/** Standard deviation of array elements */
export function std(
  a: NDArrayCore,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.std(s, ax, ddof, kd)
    ) as NDArrayCore | number;
  }
  const result = reductionOps.std(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Median of array elements */
export function median(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.median(s, ax, kd)
    ) as NDArrayCore | number;
  }
  const result = reductionOps.median(toStorage(a), axis, keepdims);
  if (typeof result === 'number') return result;
  if (result && typeof result === 'object' && 're' in result)
    return result as unknown as NDArrayCore | number;
  return fromStorage(result as ArrayStorage);
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
export function all(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | boolean {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.all(s, ax, kd)
    ) as NDArrayCore | boolean;
  }
  const result = reductionOps.all(toStorage(a), axis, keepdims);
  if (typeof result === 'boolean') return result;
  return fromStorage(result);
}

/** Test if any element is true */
export function any(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | boolean {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.any(s, ax, kd)
    ) as NDArrayCore | boolean;
  }
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
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nansum(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
  const result = reductionOps.nansum(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Product ignoring NaN */
export function nanprod(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanprod(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
  const result = reductionOps.nanprod(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Mean ignoring NaN */
export function nanmean(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanmean(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
  const result = reductionOps.nanmean(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Variance ignoring NaN */
export function nanvar(
  a: NDArrayCore,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanvar(s, ax, ddof, kd)
    ) as NDArrayCore | number;
  }
  const result = reductionOps.nanvar(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Standard deviation ignoring NaN */
export function nanstd(
  a: NDArrayCore,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanstd(s, ax, ddof, kd)
    ) as NDArrayCore | number;
  }
  const result = reductionOps.nanstd(toStorage(a), axis, ddof, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Minimum ignoring NaN */
export function nanmin(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanmin(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
  const result = reductionOps.nanmin(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) return result;
  return fromStorage(result);
}

/** Maximum ignoring NaN */
export function nanmax(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanmax(s, ax, kd)
    ) as NDArrayCore | number | Complex;
  }
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
  if (result && typeof result === 'object' && 're' in result)
    return result as unknown as NDArrayCore | number;
  return fromStorage(result as ArrayStorage);
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
