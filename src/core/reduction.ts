/**
 * Reduction Functions - Tree-shakeable standalone functions
 *
 * This module provides array reduction and aggregation functions that can be
 * imported independently for optimal tree-shaking.
 */

import type { DType } from '../common/dtype';
import * as arithmeticOps from '../common/ops/arithmetic';
import * as reductionOps from '../common/ops/reduction';
import * as shapeOps from '../common/ops/shape';
import * as sortingOps from '../common/ops/sorting';
import { asarray } from './creation';
import {
  type ArrayLike,
  ArrayStorage,
  Complex,
  fromStorage,
  type NDArrayCore,
  toStorage,
} from './types';

// ============================================================
// Reduction options (where=, initial=, dtype=)
// ============================================================

export interface ReductionOpts {
  /** Boolean mask. Only elements where `where` is True are included in the reduction. */
  where?: ArrayLike;
  /** Starting value for the reduction (combined with the result per-op). */
  initial?: number | bigint | boolean;
  /** Force input cast to this dtype before reducing. */
  dtype?: DType;
}

type ReductionKind = 'sum' | 'prod' | 'max' | 'min' | 'all' | 'any';

/** Per-op identity element used to mask out non-`where` positions. */
function identityFor(kind: ReductionKind, dtype: DType): number | bigint | boolean {
  switch (kind) {
    case 'sum':
      return dtype.startsWith('int') || dtype.startsWith('uint') ? 0 : 0;
    case 'prod':
      return 1;
    case 'max':
      return dtype === 'float64' || dtype === 'float32' || dtype === 'float16'
        ? -Infinity
        : Number.MIN_SAFE_INTEGER;
    case 'min':
      return dtype === 'float64' || dtype === 'float32' || dtype === 'float16'
        ? Infinity
        : Number.MAX_SAFE_INTEGER;
    case 'all':
      return true;
    case 'any':
      return false;
  }
}

/** Apply dtype cast and `where` mask to the input. Returns transformed storage. */
function prepareReductionInput(
  a: NDArrayCore,
  kind: ReductionKind,
  opts?: ReductionOpts,
): ArrayStorage {
  let storage = opts?.dtype ? toStorage(a.astype(opts.dtype)) : toStorage(a);
  if (opts?.where !== undefined) {
    const maskStorage = toStorage(asarray(opts.where));
    const identity = identityFor(kind, storage.dtype as DType);
    const fillStorage = ArrayStorage.empty(Array.from(storage.shape), storage.dtype);
    if (typeof identity === 'boolean') {
      for (let i = 0; i < fillStorage.size; i++) fillStorage.iset(i, identity ? 1 : 0);
    } else {
      for (let i = 0; i < fillStorage.size; i++) fillStorage.iset(i, identity as number | bigint);
    }
    storage = sortingOps.where(maskStorage, storage, fillStorage) as ArrayStorage;
  }
  return storage;
}

/** Combine the reduction result with `initial` using the op's combiner. */
function combineWithInitial(
  result: NDArrayCore | number | bigint | boolean | Complex,
  initial: number | bigint | boolean,
  kind: ReductionKind,
): NDArrayCore | number | bigint | boolean | Complex {
  if (result instanceof Complex) {
    // Complex + scalar initial: only sum/prod meaningful here.
    if (kind === 'sum') return new Complex(result.re + Number(initial), result.im);
    if (kind === 'prod') {
      const k = Number(initial);
      return new Complex(result.re * k, result.im * k);
    }
    return result;
  }

  if (typeof result === 'number' || typeof result === 'bigint' || typeof result === 'boolean') {
    const r = Number(result);
    const k = Number(initial);
    switch (kind) {
      case 'sum':
        return typeof result === 'bigint' ? result + BigInt(k) : r + k;
      case 'prod':
        return typeof result === 'bigint' ? result * BigInt(k) : r * k;
      case 'max':
        return Math.max(r, k);
      case 'min':
        return Math.min(r, k);
      case 'all':
        return Boolean(r) && Boolean(initial);
      case 'any':
        return Boolean(r) || Boolean(initial);
    }
  }

  // NDArrayCore result — combine elementwise via core arithmetic ops.
  const resStorage = toStorage(result);
  const k = Number(initial);
  switch (kind) {
    case 'sum':
      return fromStorage(arithmeticOps.add(resStorage, k));
    case 'prod':
      return fromStorage(arithmeticOps.multiply(resStorage, k));
    case 'max':
      return fromStorage(arithmeticOps.maximum(resStorage, k));
    case 'min':
      return fromStorage(arithmeticOps.minimum(resStorage, k));
    case 'all':
      // result && Boolean(initial): if initial is falsy, result becomes all-false.
      if (!initial) {
        const out = ArrayStorage.zeros(Array.from(resStorage.shape), resStorage.dtype);
        return fromStorage(out);
      }
      return fromStorage(resStorage);
    case 'any':
      // result || Boolean(initial): if initial is truthy, result becomes all-true.
      if (initial) {
        const out = ArrayStorage.empty(Array.from(resStorage.shape), resStorage.dtype);
        for (let i = 0; i < out.size; i++) out.iset(i, 1);
        return fromStorage(out);
      }
      return fromStorage(resStorage);
  }
}

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
  reduceFn: (storage: ArrayStorage, axis: number, keepdims: boolean) => ArrayStorage | T,
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
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | bigint | Complex {
  const input = fromStorage(prepareReductionInput(a, 'sum', opts));
  let result: NDArrayCore | number | bigint | Complex;
  if (Array.isArray(axis)) {
    result = reduceMultiAxis(input, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.sum(s, ax, kd),
    ) as NDArrayCore | number | bigint | Complex;
  } else {
    const r = reductionOps.sum(toStorage(input), axis, keepdims);
    result =
      typeof r === 'number' || typeof r === 'bigint' || r instanceof Complex ? r : fromStorage(r);
  }
  if (opts?.initial !== undefined) {
    return combineWithInitial(result, opts.initial, 'sum') as
      | NDArrayCore
      | number
      | bigint
      | Complex;
  }
  return result;
}

/** Mean of array elements */
export function mean(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.mean(s, ax, kd),
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
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | bigint | Complex {
  const input = fromStorage(prepareReductionInput(a, 'prod', opts));
  let result: NDArrayCore | number | bigint | Complex;
  if (Array.isArray(axis)) {
    result = reduceMultiAxis(input, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.prod(s, ax, kd),
    ) as NDArrayCore | number | bigint | Complex;
  } else {
    const r = reductionOps.prod(toStorage(input), axis, keepdims);
    result =
      typeof r === 'number' || typeof r === 'bigint' || r instanceof Complex ? r : fromStorage(r);
  }
  if (opts?.initial !== undefined) {
    return combineWithInitial(result, opts.initial, 'prod') as
      | NDArrayCore
      | number
      | bigint
      | Complex;
  }
  return result;
}

/** Maximum of array elements */
export function max(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | Complex {
  const input = fromStorage(prepareReductionInput(a, 'max', opts));
  let result: NDArrayCore | number | Complex;
  if (Array.isArray(axis)) {
    result = reduceMultiAxis(input, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.max(s, ax, kd),
    ) as NDArrayCore | number | Complex;
  } else {
    const r = reductionOps.max(toStorage(input), axis, keepdims);
    result = typeof r === 'number' || r instanceof Complex ? r : fromStorage(r);
  }
  if (opts?.initial !== undefined) {
    return combineWithInitial(result, opts.initial, 'max') as NDArrayCore | number | Complex;
  }
  return result;
}

/** Alias for max */
export const amax = max;

/** Minimum of array elements */
export function min(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | Complex {
  const input = fromStorage(prepareReductionInput(a, 'min', opts));
  let result: NDArrayCore | number | Complex;
  if (Array.isArray(axis)) {
    result = reduceMultiAxis(input, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.min(s, ax, kd),
    ) as NDArrayCore | number | Complex;
  } else {
    const r = reductionOps.min(toStorage(input), axis, keepdims);
    result = typeof r === 'number' || r instanceof Complex ? r : fromStorage(r);
  }
  if (opts?.initial !== undefined) {
    return combineWithInitial(result, opts.initial, 'min') as NDArrayCore | number | Complex;
  }
  return result;
}

/** Alias for min */
export const amin = min;

/** Peak-to-peak (max - min) */
export function ptp(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    // ptp does not compose under repeated reduction; compute as max - min across the axes.
    const maxResult = reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.max(s, ax, kd),
    );
    const minResult = reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.min(s, ax, kd),
    );
    if (typeof maxResult === 'number' && typeof minResult === 'number') {
      return maxResult - minResult;
    }
    if (maxResult instanceof Complex || minResult instanceof Complex) {
      throw new Error('multi-axis ptp not supported for complex');
    }
    return fromStorage(
      arithmeticOps.subtract(
        toStorage(maxResult as NDArrayCore),
        toStorage(minResult as NDArrayCore),
      ),
    );
  }
  const result = reductionOps.ptp(toStorage(a), axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

// ============================================================
// Index of Extrema
// ============================================================

/**
 * Re-introduce a size-1 dimension to an arg-reduction result, matching
 * NumPy's `keepdims=True` behavior introduced in 1.22.
 */
function applyArgKeepdims(
  result: ArrayStorage | number,
  a: NDArrayCore,
  axis: number | undefined,
): NDArrayCore {
  const ndim = toStorage(a).ndim;
  if (typeof result === 'number') {
    // Full reduction → shape of all 1s.
    const onesShape = Array<number>(ndim).fill(1);
    const out = ArrayStorage.zeros(onesShape, 'int64');
    out.iset(0, BigInt(result));
    return fromStorage(out);
  }
  const normAxis = axis === undefined ? 0 : axis < 0 ? ndim + axis : axis;
  return fromStorage(shapeOps.expandDims(result, normAxis));
}

/** Index of minimum value */
export function argmin(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | number {
  const result = reductionOps.argmin(toStorage(a), axis);
  if (keepdims) return applyArgKeepdims(result, a, axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Index of maximum value */
export function argmax(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | number {
  const result = reductionOps.argmax(toStorage(a), axis);
  if (keepdims) return applyArgKeepdims(result, a, axis);
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
  keepdims?: boolean,
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.variance(s, ax, ddof, kd),
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
  keepdims?: boolean,
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.std(s, ax, ddof, kd),
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
  keepdims?: boolean,
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.median(s, ax, kd),
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
  keepdims?: boolean,
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
  keepdims?: boolean,
): NDArrayCore | number {
  const result = reductionOps.quantile(toStorage(a), q, axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/**
 * Weighted average. When `returned` is true, returns `[avg, sum_of_weights]`
 * (matching `np.average(..., returned=True)`); otherwise returns just `avg`.
 */
export function average(
  a: NDArrayCore,
  axis?: number,
  weights?: NDArrayCore,
  keepdims?: boolean,
  returned?: boolean,
): NDArrayCore | number | Complex | [NDArrayCore | number | Complex, NDArrayCore | number] {
  const weightsStorage = weights ? toStorage(weights) : undefined;
  const rawAvg = reductionOps.average(toStorage(a), axis, weightsStorage, keepdims);
  const avg: NDArrayCore | number | Complex =
    typeof rawAvg === 'number' || rawAvg instanceof Complex ? rawAvg : fromStorage(rawAvg);

  if (!returned) return avg;

  // sum_of_weights: sum of weights along axis when given, else element count per slice.
  let sumOfWeights: NDArrayCore | number;
  if (weights) {
    const sw = reductionOps.sum(toStorage(weights), axis, keepdims);
    sumOfWeights =
      typeof sw === 'number' || typeof sw === 'bigint'
        ? Number(sw)
        : sw instanceof Complex
          ? sw.re
          : fromStorage(sw);
  } else {
    const aStorage = toStorage(a);
    if (axis === undefined) {
      sumOfWeights = aStorage.size;
    } else {
      const ndim = aStorage.ndim;
      const normAxis = axis < 0 ? ndim + axis : axis;
      const count = aStorage.shape[normAxis]!;
      // Build a result-shaped array filled with `count`.
      const outShape = Array.from(aStorage.shape);
      if (keepdims) outShape[normAxis] = 1;
      else outShape.splice(normAxis, 1);
      const fill = ArrayStorage.empty(outShape, 'float64');
      for (let i = 0; i < fill.size; i++) fill.iset(i, count);
      sumOfWeights = fromStorage(fill);
    }
  }

  return [avg, sumOfWeights];
}

// ============================================================
// Logical Reductions
// ============================================================

/** Test if all elements are true */
export function all(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | boolean {
  const input = fromStorage(prepareReductionInput(a, 'all', opts));
  let result: NDArrayCore | boolean;
  if (Array.isArray(axis)) {
    result = reduceMultiAxis(input, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.all(s, ax, kd),
    ) as NDArrayCore | boolean;
  } else {
    const r = reductionOps.all(toStorage(input), axis, keepdims);
    result = typeof r === 'boolean' ? r : fromStorage(r);
  }
  if (opts?.initial !== undefined) {
    return combineWithInitial(result, opts.initial, 'all') as NDArrayCore | boolean;
  }
  return result;
}

/** Test if any element is true */
export function any(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | boolean {
  const input = fromStorage(prepareReductionInput(a, 'any', opts));
  let result: NDArrayCore | boolean;
  if (Array.isArray(axis)) {
    result = reduceMultiAxis(input, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.any(s, ax, kd),
    ) as NDArrayCore | boolean;
  } else {
    const r = reductionOps.any(toStorage(input), axis, keepdims);
    result = typeof r === 'boolean' ? r : fromStorage(r);
  }
  if (opts?.initial !== undefined) {
    return combineWithInitial(result, opts.initial, 'any') as NDArrayCore | boolean;
  }
  return result;
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
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nansum(s, ax, kd),
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
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanprod(s, ax, kd),
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
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanmean(s, ax, kd),
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
  keepdims?: boolean,
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanvar(s, ax, ddof, kd),
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
  keepdims?: boolean,
): NDArrayCore | number {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanstd(s, ax, ddof, kd),
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
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanmin(s, ax, kd),
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
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  if (Array.isArray(axis)) {
    return reduceMultiAxis(a, axis, keepdims ?? false, (s, ax, kd) =>
      reductionOps.nanmax(s, ax, kd),
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
  keepdims?: boolean,
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
  keepdims?: boolean,
): NDArrayCore | number {
  const result = reductionOps.nanpercentile(toStorage(a), q, axis, keepdims);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}
