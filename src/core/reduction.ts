/**
 * Reduction Functions - Tree-shakeable standalone functions
 *
 * This module provides array reduction and aggregation functions that can be
 * imported independently for optimal tree-shaking.
 */

import {
  type DType,
  isComplexDType,
  isFloatDType,
  promoteDTypes,
  throwIfComplex,
  trueDivideResultDtype,
} from '../common/dtype';
import * as arithmeticOps from '../common/ops/arithmetic';
import * as reductionOps from '../common/ops/reduction';
import * as shapeOps from '../common/ops/shape';
import * as sortingOps from '../common/ops/sorting';
import { checkAxis, notAnNDArray } from '../common/storage';
import { asarray } from './creation';
import {
  type ArrayLike,
  ArrayStorage,
  Complex,
  fromStorage,
  NDArrayCore,
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
// Axis dispatch
// ============================================================

type ReduceFn<T> = (
  s: ArrayStorage,
  axis: number | undefined,
  keepdims: boolean,
) => ArrayStorage | T;

function onesShape(ndim: number): number[] {
  return Array<number>(ndim).fill(1);
}

/**
 * Reduce along multiple axes by repeatedly calling a single-axis reducer. Only valid for
 * reductions that compose (sum, prod, max, min, all, any and their nan variants).
 * Axes are processed in descending order so removing one does not shift the rest.
 */
function reduceMultiAxis<T>(
  s: ArrayStorage,
  axes: number[],
  keepdims: boolean,
  reduceFn: ReduceFn<T>,
): NDArrayCore | T {
  const ndim = s.ndim;
  const normAxes = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  const sortedDesc = [...normAxes].sort((x, y) => y - x);
  const sortedAsc = [...normAxes].sort((x, y) => x - y);

  let current = s;
  for (const ax of sortedDesc) {
    const r = reduceFn(current, ax, false);
    if (!(r instanceof ArrayStorage)) {
      if (keepdims) {
        const out = ArrayStorage.zeros(onesShape(ndim), current.dtype);
        out.iset(0, r as number | bigint | Complex);
        return fromStorage(out);
      }
      return r as T;
    }
    current = r;
  }

  if (keepdims) {
    for (const ax of sortedAsc) {
      current = shapeOps.expandDims(current, ax);
    }
  }
  return fromStorage(current);
}

/**
 * Reduce several axes at once by moving them last and merging them into one axis. Needed for
 * reductions that do not compose across axes (std, var, median, nanmean and similar), where
 * reducing one axis at a time gives a different answer.
 */
function reduceJointAxes<T>(
  s: ArrayStorage,
  axes: number[],
  keepdims: boolean,
  reduceFn: ReduceFn<T>,
): NDArrayCore | T {
  const ndim = s.ndim;
  const norm = axes.map((ax) => {
    const n = ax < 0 ? ndim + ax : ax;
    if (n < 0 || n >= ndim) {
      throw new Error(`axis ${ax} is out of bounds for array of dimension ${ndim}`);
    }
    return n;
  });
  if (new Set(norm).size !== norm.length) throw new Error("duplicate value in 'axis'");
  if (norm.length === ndim) return reduceAxes(s, undefined, keepdims, false, reduceFn);

  const sortedAsc = [...norm].sort((x, y) => x - y);
  const kept: number[] = [];
  for (let i = 0; i < ndim; i++) if (!norm.includes(i)) kept.push(i);
  const merged = shapeOps.reshape(shapeOps.transpose(s, [...kept, ...sortedAsc]), [
    ...kept.map((i) => s.shape[i]!),
    sortedAsc.reduce((n, i) => n * s.shape[i]!, 1),
  ]);
  // At least one axis is kept, so the reduction leaves an array.
  let out = reduceFn(merged, merged.ndim - 1, false) as ArrayStorage;
  if (keepdims) {
    for (const ax of sortedAsc) out = shapeOps.expandDims(out, ax);
  }
  return fromStorage(out);
}

/**
 * Entry point for every reduction with an axis argument. Validates the axis, maps null to
 * undefined, and routes the three shapes of call: a tuple of axes, no axis with keepdims, and
 * a single axis (or none). No axis with keepdims reduces the flattened array along axis 0 so
 * the result keeps the dtype the single-axis path produces, then restores one size-1 dim per
 * input dimension, as NumPy does. A 0-d input has no axis to keep and returns the plain scalar.
 */
function reduceAxes<T>(
  s: ArrayStorage,
  axis: unknown,
  keepdims: boolean | undefined,
  allowTuple: boolean,
  reduceFn: ReduceFn<T>,
  joint = false,
): NDArrayCore | T {
  const ax = checkAxis(axis, allowTuple);
  const kd = keepdims ?? false;
  if (Array.isArray(ax)) {
    return joint ? reduceJointAxes(s, ax, kd, reduceFn) : reduceMultiAxis(s, ax, kd, reduceFn);
  }
  if (ax === undefined && kd && s.ndim > 0) {
    const r = reduceFn(shapeOps.ravel(s), 0, true) as ArrayStorage;
    return fromStorage(shapeOps.reshape(r, onesShape(s.ndim)));
  }
  const r = reduceFn(s, ax, kd);
  return r instanceof ArrayStorage ? fromStorage(r) : r;
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
  const input = prepareReductionInput(a, 'sum', opts);
  const result = reduceAxes(input, axis, keepdims, true, (s, ax, kd) =>
    reductionOps.sum(s, ax, kd),
  );
  return opts?.initial !== undefined
    ? (combineWithInitial(result, opts.initial, 'sum') as NDArrayCore | number | bigint | Complex)
    : result;
}

/** Mean of array elements */
export function mean(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | bigint | Complex {
  return reduceAxes(toStorage(a), axis, keepdims, true, (s, ax, kd) =>
    reductionOps.mean(s, ax, kd),
  );
}

/** Product of array elements */
export function prod(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | bigint | Complex {
  const input = prepareReductionInput(a, 'prod', opts);
  const result = reduceAxes(input, axis, keepdims, true, (s, ax, kd) =>
    reductionOps.prod(s, ax, kd),
  );
  return opts?.initial !== undefined
    ? (combineWithInitial(result, opts.initial, 'prod') as NDArrayCore | number | bigint | Complex)
    : result;
}

/** Maximum of array elements */
export function max(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | bigint | Complex {
  const input = prepareReductionInput(a, 'max', opts);
  const result = reduceAxes(input, axis, keepdims, true, (s, ax, kd) =>
    reductionOps.max(s, ax, kd),
  );
  return opts?.initial !== undefined
    ? (combineWithInitial(result, opts.initial, 'max') as NDArrayCore | number | bigint | Complex)
    : result;
}

/** Alias for max */
export const amax = max;

/** Minimum of array elements */
export function min(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | number | bigint | Complex {
  const input = prepareReductionInput(a, 'min', opts);
  const result = reduceAxes(input, axis, keepdims, true, (s, ax, kd) =>
    reductionOps.min(s, ax, kd),
  );
  return opts?.initial !== undefined
    ? (combineWithInitial(result, opts.initial, 'min') as NDArrayCore | number | bigint | Complex)
    : result;
}

/** Alias for min */
export const amin = min;

/** Peak-to-peak (max - min) */
export function ptp(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | bigint | Complex {
  const ax = checkAxis(axis, true);
  const s = toStorage(a);
  if (!Array.isArray(ax)) {
    return reduceAxes(s, ax, keepdims, true, (st, x, kd) => reductionOps.ptp(st, x, kd));
  }
  // ptp does not compose under repeated reduction; compute as max - min across the axes.
  const kd = keepdims ?? false;
  const maxResult = reduceMultiAxis(s, ax, kd, (st, x, k) => reductionOps.max(st, x as number, k));
  const minResult = reduceMultiAxis(s, ax, kd, (st, x, k) => reductionOps.min(st, x as number, k));
  if (typeof maxResult === 'number' && typeof minResult === 'number') return maxResult - minResult;
  if (typeof maxResult === 'bigint' && typeof minResult === 'bigint') return maxResult - minResult;
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

// ============================================================
// Index of Extrema
// ============================================================

/** Re-introduce a size-1 dimension to an arg-reduction result (NumPy 1.22 keepdims). */
function applyArgKeepdims(
  result: ArrayStorage | number,
  a: NDArrayCore,
  axis: number | undefined,
): NDArrayCore {
  const ndim = toStorage(a).ndim;
  if (typeof result === 'number') {
    const out = ArrayStorage.zeros(onesShape(ndim), 'int32');
    out.iset(0, result);
    return fromStorage(out);
  }
  const normAxis = axis === undefined ? 0 : axis < 0 ? ndim + axis : axis;
  return fromStorage(shapeOps.expandDims(result, normAxis));
}

/** Index of minimum value */
export function argmin(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | number {
  const ax = checkAxis(axis, false) as number | undefined;
  const result = reductionOps.argmin(toStorage(a), ax);
  if (keepdims) return applyArgKeepdims(result, a, ax);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Index of maximum value */
export function argmax(a: NDArrayCore, axis?: number, keepdims?: boolean): NDArrayCore | number {
  const ax = checkAxis(axis, false) as number | undefined;
  const result = reductionOps.argmax(toStorage(a), ax);
  if (keepdims) return applyArgKeepdims(result, a, ax);
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
  return reduceAxes(
    toStorage(a),
    axis,
    keepdims,
    true,
    (s, ax, kd) => reductionOps.variance(s, ax, ddof, kd),
    true,
  );
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
  return reduceAxes(
    toStorage(a),
    axis,
    keepdims,
    true,
    (s, ax, kd) => reductionOps.std(s, ax, ddof, kd),
    true,
  );
}

/** Median of array elements */
export function median(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  return reduceAxes(
    toStorage(a),
    axis,
    keepdims,
    true,
    (s, ax, kd) => reductionOps.median(s, ax, kd),
    true,
  );
}

type QuantileOp = (
  s: ArrayStorage,
  q: number,
  axis: number | undefined,
  keepdims: boolean,
) => ArrayStorage | number;

/**
 * Shared body of percentile, quantile and their nan variants. A scalar q reduces directly. An
 * NDArray q gives q.shape followed by the reduced shape, one reduction per q value.
 */
function quantileReduce(
  a: NDArrayCore,
  q: number | NDArrayCore,
  axis: number | undefined,
  keepdims: boolean | undefined,
  op: QuantileOp,
): NDArrayCore | number {
  const reduceOne = (s: ArrayStorage, qv: number) =>
    reduceAxes(s, axis, keepdims, false, (st, ax, kd) => op(st, qv, ax as number | undefined, kd));
  if (typeof q === 'number') return reduceOne(toStorage(a), q);
  if (!(q instanceof NDArrayCore)) {
    // Complex data is reported ahead of a bad q, as the scalar path does.
    throwIfComplex(a.dtype, 'quantile', 'Complex numbers are not orderable.');
    throw notAnNDArray(q);
  }

  const qs = toStorage(q);
  if (isComplexDType(qs.dtype)) throw new TypeError('q is not supported for complex numbers.');
  if (qs.ndim > 2) throw new Error('q must be a scalar or 1d');
  // The reduction narrows its result to the data dtype, so float32 data with float64 q is widened
  // first to keep the extra precision.
  const outDtype = promoteDTypes(trueDivideResultDtype(a.dtype), trueDivideResultDtype(qs.dtype));
  const data = outDtype === 'float64' && isFloatDType(a.dtype) ? a.astype('float64') : a;
  const s = toStorage(data);

  const results: (ArrayStorage | number)[] = [];
  for (let i = 0; i < qs.size; i++) {
    const r = reduceOne(s, Number(qs.iget(i)));
    results.push(r instanceof NDArrayCore ? toStorage(r) : r);
  }
  const probe = results[0] ?? (reduceOne(s, 0.5) as NDArrayCore | number);
  const first = probe instanceof NDArrayCore ? toStorage(probe) : probe;
  const innerShape = first instanceof ArrayStorage ? Array.from(first.shape) : [];
  return fromStorage(reductionOps.stackQuantileResults(s, qs, results, innerShape));
}

/**
 * Percentile of array elements. An NDArray q gives an array of shape q.shape followed by the
 * reduced shape.
 */
export function percentile(
  a: NDArrayCore,
  q: number | NDArrayCore,
  axis?: number,
  keepdims?: boolean,
): NDArrayCore | number {
  return quantileReduce(a, q, axis, keepdims, reductionOps.percentile);
}

/**
 * Quantile of array elements. An NDArray q gives an array of shape q.shape followed by the
 * reduced shape.
 */
export function quantile(
  a: NDArrayCore,
  q: number | NDArrayCore,
  axis?: number,
  keepdims?: boolean,
): NDArrayCore | number {
  return quantileReduce(a, q, axis, keepdims, reductionOps.quantile);
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
):
  | NDArrayCore
  | number
  | bigint
  | Complex
  | [NDArrayCore | number | Complex, NDArrayCore | number] {
  const ax = checkAxis(axis, false) as number | undefined;
  const aStorage = toStorage(a);
  const weightsStorage = weights ? toStorage(weights) : undefined;
  // With keepdims and no axis, both operands are flattened and reduced along axis 0.
  const flat = ax === undefined && keepdims && aStorage.ndim > 0;
  const flatWeights =
    flat && weightsStorage && weightsStorage.size === aStorage.size
      ? shapeOps.ravel(weightsStorage)
      : weightsStorage;
  const restore = (r: ArrayStorage): NDArrayCore =>
    fromStorage(flat ? shapeOps.reshape(r, onesShape(aStorage.ndim)) : r);

  const rawAvg = flat
    ? reductionOps.average(shapeOps.ravel(aStorage), 0, flatWeights, true)
    : reductionOps.average(aStorage, ax, weightsStorage, keepdims);
  const avg: NDArrayCore | number | Complex =
    rawAvg instanceof ArrayStorage ? restore(rawAvg) : rawAvg;

  if (!returned) return avg;

  // sum_of_weights: sum of weights along axis when given, else element count per slice.
  let sumOfWeights: NDArrayCore | number;
  if (weightsStorage) {
    const sw = flat
      ? reductionOps.sum(flatWeights!, 0, true)
      : reductionOps.sum(weightsStorage, ax, keepdims);
    sumOfWeights =
      typeof sw === 'number' || typeof sw === 'bigint'
        ? Number(sw)
        : sw instanceof Complex
          ? sw.re
          : restore(sw);
  } else if (ax === undefined) {
    if (flat) {
      const fill = ArrayStorage.empty(onesShape(aStorage.ndim), 'float64');
      fill.iset(0, aStorage.size);
      sumOfWeights = fromStorage(fill);
    } else {
      sumOfWeights = aStorage.size;
    }
  } else {
    const ndim = aStorage.ndim;
    const normAxis = ax < 0 ? ndim + ax : ax;
    const count = aStorage.shape[normAxis]!;
    // Build a result-shaped array filled with `count`.
    const outShape = Array.from(aStorage.shape);
    if (keepdims) outShape[normAxis] = 1;
    else outShape.splice(normAxis, 1);
    const fill = ArrayStorage.empty(outShape, 'float64');
    for (let i = 0; i < fill.size; i++) fill.iset(i, count);
    sumOfWeights = fromStorage(fill);
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
  const input = prepareReductionInput(a, 'all', opts);
  const result = reduceAxes(input, axis, keepdims, true, (s, ax, kd) =>
    reductionOps.all(s, ax, kd),
  );
  return opts?.initial !== undefined
    ? (combineWithInitial(result, opts.initial, 'all') as NDArrayCore | boolean)
    : result;
}

/** Test if any element is true */
export function any(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
  opts?: ReductionOpts,
): NDArrayCore | boolean {
  const input = prepareReductionInput(a, 'any', opts);
  const result = reduceAxes(input, axis, keepdims, true, (s, ax, kd) =>
    reductionOps.any(s, ax, kd),
  );
  return opts?.initial !== undefined
    ? (combineWithInitial(result, opts.initial, 'any') as NDArrayCore | boolean)
    : result;
}

// ============================================================
// Cumulative Functions
// ============================================================

/** Cumulative sum */
export function cumsum(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(
    reductionOps.cumsum(toStorage(a), checkAxis(axis, false) as number | undefined),
  );
}

/** Alias for cumsum */
export const cumulative_sum = cumsum;

/** Cumulative product */
export function cumprod(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(
    reductionOps.cumprod(toStorage(a), checkAxis(axis, false) as number | undefined),
  );
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
): NDArrayCore | number | bigint | Complex {
  return reduceAxes(toStorage(a), axis, keepdims, true, (s, ax, kd) =>
    reductionOps.nansum(s, ax, kd),
  );
}

/** Product ignoring NaN */
export function nanprod(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | bigint | Complex {
  return reduceAxes(toStorage(a), axis, keepdims, true, (s, ax, kd) =>
    reductionOps.nanprod(s, ax, kd),
  );
}

/** Mean ignoring NaN */
export function nanmean(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | bigint | Complex {
  return reduceAxes(
    toStorage(a),
    axis,
    keepdims,
    true,
    (s, ax, kd) => reductionOps.nanmean(s, ax, kd),
    true,
  );
}

/** Variance ignoring NaN */
export function nanvar(
  a: NDArrayCore,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean,
): NDArrayCore | number {
  return reduceAxes(
    toStorage(a),
    axis,
    keepdims,
    true,
    (s, ax, kd) => reductionOps.nanvar(s, ax, ddof, kd),
    true,
  );
}

/** Standard deviation ignoring NaN */
export function nanstd(
  a: NDArrayCore,
  axis?: number | number[],
  ddof?: number,
  keepdims?: boolean,
): NDArrayCore | number {
  return reduceAxes(
    toStorage(a),
    axis,
    keepdims,
    true,
    (s, ax, kd) => reductionOps.nanstd(s, ax, ddof, kd),
    true,
  );
}

/** Minimum ignoring NaN */
export function nanmin(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | bigint | Complex {
  return reduceAxes(toStorage(a), axis, keepdims, true, (s, ax, kd) =>
    reductionOps.nanmin(s, ax, kd),
  );
}

/** Maximum ignoring NaN */
export function nanmax(
  a: NDArrayCore,
  axis?: number | number[],
  keepdims?: boolean,
): NDArrayCore | number | bigint | Complex {
  return reduceAxes(toStorage(a), axis, keepdims, true, (s, ax, kd) =>
    reductionOps.nanmax(s, ax, kd),
  );
}

/** Index of minimum ignoring NaN */
export function nanargmin(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = reductionOps.nanargmin(toStorage(a), checkAxis(axis, false) as number | undefined);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Index of maximum ignoring NaN */
export function nanargmax(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = reductionOps.nanargmax(toStorage(a), checkAxis(axis, false) as number | undefined);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}

/** Cumulative sum ignoring NaN */
export function nancumsum(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(
    reductionOps.nancumsum(toStorage(a), checkAxis(axis, false) as number | undefined),
  );
}

/** Cumulative product ignoring NaN */
export function nancumprod(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(
    reductionOps.nancumprod(toStorage(a), checkAxis(axis, false) as number | undefined),
  );
}

/** Median ignoring NaN */
export function nanmedian(
  a: NDArrayCore,
  axis?: number,
  keepdims?: boolean,
): NDArrayCore | number | Complex {
  return reduceAxes(toStorage(a), axis, keepdims, false, (s, ax, kd) =>
    reductionOps.nanmedian(s, ax as number | undefined, kd),
  );
}

/** Quantile ignoring NaN. An NDArray q gives q.shape followed by the reduced shape. */
export function nanquantile(
  a: NDArrayCore,
  q: number | NDArrayCore,
  axis?: number,
  keepdims?: boolean,
): NDArrayCore | number {
  return quantileReduce(a, q, axis, keepdims, reductionOps.nanquantile);
}

/** Percentile ignoring NaN. An NDArray q gives q.shape followed by the reduced shape. */
export function nanpercentile(
  a: NDArrayCore,
  q: number | NDArrayCore,
  axis?: number,
  keepdims?: boolean,
): NDArrayCore | number {
  return quantileReduce(a, q, axis, keepdims, reductionOps.nanpercentile);
}
