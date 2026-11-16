/**
 * Sorting and Searching Functions - Tree-shakeable standalone functions
 *
 * This module provides sorting, searching, and counting functions that can be
 * imported independently for optimal tree-shaking.
 */

import { type DType, isComplexDType, isFloatDType, isIntegerDType } from '../common/dtype';
import { normalizeAxisTuple } from '../common/internal/indexing';
import * as reductionOps from '../common/ops/reduction';
import * as sortingOps from '../common/ops/sorting';
import { checkAxis } from '../common/storage';
import { asarray } from './creation';
import {
  type ArrayLike,
  type ArrayStorage,
  fromStorage,
  fromStorageArray,
  type NDArrayCore,
  toStorage,
} from './types';

// ============================================================
// Sorting
// ============================================================

/** Sort array along axis */
export function sort(a: NDArrayCore, axis: number = -1): NDArrayCore {
  return fromStorage(sortingOps.sort(toStorage(a), axis));
}

/** Indices that would sort array */
export function argsort(a: NDArrayCore, axis: number = -1): NDArrayCore {
  return fromStorage(sortingOps.argsort(toStorage(a), axis));
}

/** Indirect stable sort on multiple keys */
export function lexsort(keys: NDArrayCore[]): NDArrayCore {
  return fromStorage(sortingOps.lexsort(keys.map((k) => toStorage(k))));
}

/** Partially sort array */
export function partition(a: NDArrayCore, kth: number, axis: number = -1): NDArrayCore {
  return fromStorage(sortingOps.partition(toStorage(a), kth, axis));
}

/** Indices that would partially sort array */
export function argpartition(a: NDArrayCore, kth: number, axis: number = -1): NDArrayCore {
  return fromStorage(sortingOps.argpartition(toStorage(a), kth, axis));
}

/** Sort complex array */
export function sort_complex(a: NDArrayCore): NDArrayCore {
  return fromStorage(sortingOps.sort_complex(toStorage(a)));
}

// ============================================================
// Searching
// ============================================================

/** Indices of non-zero elements */
export function nonzero(a: NDArrayCore): NDArrayCore[] {
  return fromStorageArray(sortingOps.nonzero(toStorage(a)));
}

/** Indices where condition is True */
export function argwhere(a: NDArrayCore): NDArrayCore {
  return fromStorage(sortingOps.argwhere(toStorage(a)));
}

/** Indices of non-zero elements in flattened array */
export function flatnonzero(a: NDArrayCore): NDArrayCore {
  return fromStorage(sortingOps.flatnonzero(toStorage(a)));
}

/** Return elements from x or y based on condition, or the indices where condition is true when x and y are omitted */
export function where(condition: NDArrayCore): NDArrayCore[];
export function where(condition: NDArrayCore, x: ArrayLike, y: ArrayLike): NDArrayCore;
export function where(
  condition: NDArrayCore,
  x?: ArrayLike,
  y?: ArrayLike,
): NDArrayCore | NDArrayCore[];
export function where(
  condition: NDArrayCore,
  x?: ArrayLike,
  y?: ArrayLike,
): NDArrayCore | NDArrayCore[] {
  let xArr = x === undefined ? undefined : asarray(x);
  let yArr = y === undefined ? undefined : asarray(y);
  if (xArr && yArr) {
    xArr = weakScalar(x, xArr, yArr.dtype as DType);
    yArr = weakScalar(y, yArr, xArr.dtype as DType);
  }
  const result = sortingOps.where(
    toStorage(condition),
    xArr && toStorage(xArr),
    yArr && toStorage(yArr),
  );
  if (Array.isArray(result)) {
    return fromStorageArray(result);
  }
  return fromStorage(result);
}

/**
 * Re-type a where() operand that came from a JS scalar the way NumPy treats a
 * weak scalar: it takes the other operand's dtype when that dtype can hold it,
 * so where(mask, int32Arr, 0) stays int32 rather than promoting to float64. As
 * in NumPy, an integer out of the other dtype's range wraps instead of widening.
 * A fractional value against an integer or bool array keeps the default dtype
 * it was converted with.
 *
 * @param value - The operand as passed to where()
 * @param arr - That operand already converted with asarray
 * @param other - The other operand's dtype
 * @returns arr, or the scalar re-created in the other operand's dtype
 */
function weakScalar(value: unknown, arr: NDArrayCore, other: DType): NDArrayCore {
  if (typeof value === 'boolean') return asarray(value, other);
  if (typeof value !== 'number') return arr;
  if (isFloatDType(other) || isComplexDType(other)) return asarray(value, other);
  if (isIntegerDType(other) && Number.isInteger(value)) return asarray(value, other);
  return arr;
}

/** Find indices for inserting elements to maintain order */
export function searchsorted(
  a: NDArrayCore,
  v: NDArrayCore,
  side: 'left' | 'right' = 'left',
): NDArrayCore {
  return fromStorage(sortingOps.searchsorted(toStorage(a), toStorage(v), side));
}

/** Extract elements where condition is True */
export function extract(condition: NDArrayCore, a: NDArrayCore): NDArrayCore {
  return fromStorage(sortingOps.extract(toStorage(condition), toStorage(a)));
}

// ============================================================
// Counting
// ============================================================

/** Count non-zero elements */
export function count_nonzero(a: NDArrayCore, axis?: number | number[]): NDArrayCore | number {
  const ax = checkAxis(axis, true);
  if (Array.isArray(ax)) {
    const normAxes = normalizeAxisTuple(ax, toStorage(a).ndim);
    const sortedDesc = [...normAxes].sort((x, y) => y - x);
    let current: ArrayStorage = toStorage(a);
    // Only the first axis counts non-zeros; the counts that remain are summed over the rest.
    for (const [i, d] of sortedDesc.entries()) {
      const r =
        i === 0 ? sortingOps.count_nonzero(current, d) : reductionOps.sum(current, d, false);
      if (typeof r === 'number') return r;
      current = r as ArrayStorage;
    }
    return fromStorage(current);
  }
  const result = sortingOps.count_nonzero(toStorage(a), ax);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}
