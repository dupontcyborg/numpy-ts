/**
 * Sorting and Searching Functions - Tree-shakeable standalone functions
 *
 * This module provides sorting, searching, and counting functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as sortingOps from '../ops/sorting';
import { NDArrayCore, toStorage, fromStorage, fromStorageArray } from './types';

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

/** Return elements from x or y based on condition */
export function where(
  condition: NDArrayCore,
  x?: NDArrayCore,
  y?: NDArrayCore
): NDArrayCore | NDArrayCore[] {
  const result = sortingOps.where(
    toStorage(condition),
    x ? toStorage(x) : undefined,
    y ? toStorage(y) : undefined
  );
  if (Array.isArray(result)) {
    return fromStorageArray(result);
  }
  return fromStorage(result);
}

/** Find indices for inserting elements to maintain order */
export function searchsorted(
  a: NDArrayCore,
  v: NDArrayCore,
  side: 'left' | 'right' = 'left'
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
export function count_nonzero(a: NDArrayCore, axis?: number): NDArrayCore | number {
  const result = sortingOps.count_nonzero(toStorage(a), axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}
