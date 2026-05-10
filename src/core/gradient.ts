/**
 * Gradient and Differentiation Functions - Tree-shakeable standalone functions
 *
 * This module provides differentiation operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as gradientOps from '../common/ops/gradient';
import { NDArrayCore } from '../common/ndarray-core';
import { fromStorage, fromStorageArray, toStorage } from './types';

/** Calculate n-th discrete difference */
export function diff(a: NDArrayCore, n?: number, axis?: number): NDArrayCore {
  return fromStorage(gradientOps.diff(toStorage(a), n, axis));
}

/** Difference between consecutive elements in 1D array */
export function ediff1d(
  a: NDArrayCore,
  to_end?: number[] | null,
  to_begin?: number[] | null,
): NDArrayCore {
  return fromStorage(gradientOps.ediff1d(toStorage(a), to_end ?? null, to_begin ?? null));
}

/** Per-axis spacing: scalar (uniform) or 1-D coordinate array (non-uniform). */
export type GradientSpacingArg = number | number[] | NDArrayCore;

/** Return gradient of N-dimensional array */
export function gradient(
  f: NDArrayCore,
  varargs?: number | GradientSpacingArg[],
  axis?: number | number[] | null,
): NDArrayCore | NDArrayCore[] {
  let opVarargs: number | gradientOps.GradientSpacing[] | undefined;
  if (varargs === undefined || typeof varargs === 'number') {
    opVarargs = varargs;
  } else {
    opVarargs = varargs.map((v) => {
      if (typeof v === 'number') return v;
      if (v instanceof NDArrayCore) {
        return Array.from(toStorage(v).data as ArrayLike<number>, (x) => Number(x));
      }
      return v;
    });
  }
  const result = gradientOps.gradient(toStorage(f), opVarargs, axis);
  if (Array.isArray(result)) {
    return fromStorageArray(result);
  }
  return fromStorage(result);
}
