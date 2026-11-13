/**
 * Set Operations - Tree-shakeable standalone functions
 *
 * This module provides set operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as setOps from '../common/ops/sets';
import { type ArrayStorage, fromStorage, type NDArrayCore, toStorage } from './types';

type UniqueField<K extends string, T, Flag extends boolean> = [Flag] extends [true]
  ? { [P in K]: T }
  : { [P in K]?: T };

type Flatten<T> = { [P in keyof T]: T[P] };

type UniqueObject<T, I extends boolean, V extends boolean, C extends boolean> = Flatten<
  { values: T } & UniqueField<'indices', T, I> &
    UniqueField<'inverse', T, V> &
    UniqueField<'counts', T, C>
>;

/**
 * Result of unique for the given flag types, where T is the array type. No flags gives a bare
 * array, a literal-true flag makes its field non-optional, and a flag typed as plain boolean
 * gives the union since the shape is only known at runtime.
 */
export type UniqueResult<T, I extends boolean, V extends boolean, C extends boolean> = [
  I,
  V,
  C,
] extends [false, false, false]
  ? T
  : [I] extends [true]
    ? UniqueObject<T, I, V, C>
    : [V] extends [true]
      ? UniqueObject<T, I, V, C>
      : [C] extends [true]
        ? UniqueObject<T, I, V, C>
        : T | UniqueObject<T, I, V, C>;

/**
 * Find unique elements. With no flags the result is the sorted unique array; with any flag set it
 * is an object holding values plus the requested indices, inverse and counts.
 *
 * @param ar - Input array
 * @param returnIndex - Also return the first-occurrence indices
 * @param returnInverse - Also return the indices that rebuild the input
 * @param returnCounts - Also return the count of each unique value
 * @param axis - Axis to operate along, or undefined to flatten
 * @returns Unique values, or an object when any flag is set
 */
export function unique<
  I extends boolean = false,
  V extends boolean = false,
  C extends boolean = false,
>(
  ar: NDArrayCore,
  returnIndex?: I,
  returnInverse?: V,
  returnCounts?: C,
  axis?: number,
): UniqueResult<NDArrayCore, I, V, C> {
  const result = setOps.unique(toStorage(ar), returnIndex, returnInverse, returnCounts, axis);
  if ('values' in result) {
    return {
      values: fromStorage(result.values),
      indices: result.indices ? fromStorage(result.indices) : undefined,
      inverse: result.inverse ? fromStorage(result.inverse) : undefined,
      counts: result.counts ? fromStorage(result.counts) : undefined,
    } as UniqueResult<NDArrayCore, I, V, C>;
  }
  return fromStorage(result as ArrayStorage) as UniqueResult<NDArrayCore, I, V, C>;
}

/**
 * Test if elements in ar1 are in ar2.
 * @deprecated Use {@link isin} instead. `in1d` follows NumPy's deprecation and will be removed in a future release.
 */
export function in1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.in1d(toStorage(ar1), toStorage(ar2)));
}

/** Find intersection of two arrays */
export function intersect1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.intersect1d(toStorage(ar1), toStorage(ar2)));
}

/** Test if elements are in test elements */
export function isin(element: NDArrayCore, testElements: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.isin(toStorage(element), toStorage(testElements)));
}

/** Find set difference (elements in ar1 not in ar2) */
export function setdiff1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.setdiff1d(toStorage(ar1), toStorage(ar2)));
}

/** Find symmetric difference (elements in either but not both) */
export function setxor1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.setxor1d(toStorage(ar1), toStorage(ar2)));
}

/** Find union of two arrays */
export function union1d(ar1: NDArrayCore, ar2: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.union1d(toStorage(ar1), toStorage(ar2)));
}

/** Trim leading/trailing zeros */
export function trim_zeros(filt: NDArrayCore, trim?: 'f' | 'b' | 'fb'): NDArrayCore {
  return fromStorage(setOps.trim_zeros(toStorage(filt), trim));
}

/** Return all unique values and metadata (Array API) */
export function unique_all(x: NDArrayCore): {
  values: NDArrayCore;
  indices: NDArrayCore;
  inverse_indices: NDArrayCore;
  counts: NDArrayCore;
} {
  const result = setOps.unique_all(toStorage(x));
  return {
    values: fromStorage(result.values),
    indices: fromStorage(result.indices),
    inverse_indices: fromStorage(result.inverse_indices),
    counts: fromStorage(result.counts),
  };
}

/** Return unique values and counts (Array API) */
export function unique_counts(x: NDArrayCore): {
  values: NDArrayCore;
  counts: NDArrayCore;
} {
  const result = setOps.unique_counts(toStorage(x));
  return {
    values: fromStorage(result.values),
    counts: fromStorage(result.counts),
  };
}

/** Return unique values and inverse indices (Array API) */
export function unique_inverse(x: NDArrayCore): {
  values: NDArrayCore;
  inverse_indices: NDArrayCore;
} {
  const result = setOps.unique_inverse(toStorage(x));
  return {
    values: fromStorage(result.values),
    inverse_indices: fromStorage(result.inverse_indices),
  };
}

/** Return unique values (Array API) */
export function unique_values(x: NDArrayCore): NDArrayCore {
  return fromStorage(setOps.unique_values(toStorage(x)));
}
