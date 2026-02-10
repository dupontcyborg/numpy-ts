/**
 * Set Operations - Tree-shakeable standalone functions
 *
 * This module provides set operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as setOps from '../ops/sets';
import { NDArrayCore, toStorage, fromStorage, ArrayStorage } from './types';

/** Find unique elements */
export function unique(
  ar: NDArrayCore,
  returnIndex: boolean = false,
  returnInverse: boolean = false,
  returnCounts: boolean = false
):
  | NDArrayCore
  | { values: NDArrayCore; indices?: NDArrayCore; inverse?: NDArrayCore; counts?: NDArrayCore } {
  const result = setOps.unique(toStorage(ar), returnIndex, returnInverse, returnCounts);
  if ('values' in result) {
    return {
      values: fromStorage(result.values),
      indices: result.indices ? fromStorage(result.indices) : undefined,
      inverse: result.inverse ? fromStorage(result.inverse) : undefined,
      counts: result.counts ? fromStorage(result.counts) : undefined,
    };
  }
  return fromStorage(result as ArrayStorage);
}

/** Test if elements in ar1 are in ar2 (deprecated, use isin) */
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
