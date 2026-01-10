/**
 * Gradient and Differentiation Functions - Tree-shakeable standalone functions
 *
 * This module provides differentiation operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as gradientOps from '../ops/gradient';
import { NDArrayCore, toStorage, fromStorage, fromStorageArray } from './types';

/** Calculate n-th discrete difference */
export function diff(a: NDArrayCore, n?: number, axis?: number): NDArrayCore {
  return fromStorage(gradientOps.diff(toStorage(a), n, axis));
}

/** Difference between consecutive elements in 1D array */
export function ediff1d(
  a: NDArrayCore,
  to_end?: number[] | null,
  to_begin?: number[] | null
): NDArrayCore {
  return fromStorage(gradientOps.ediff1d(toStorage(a), to_end ?? null, to_begin ?? null));
}

/** Return gradient of N-dimensional array */
export function gradient(
  f: NDArrayCore,
  varargs?: number | number[],
  axis?: number | number[] | null
): NDArrayCore | NDArrayCore[] {
  const result = gradientOps.gradient(toStorage(f), varargs, axis);
  if (Array.isArray(result)) {
    return fromStorageArray(result);
  }
  return fromStorage(result);
}
