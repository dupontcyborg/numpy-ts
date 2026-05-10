/**
 * Gradient and Differentiation Functions - Tree-shakeable standalone functions
 *
 * This module provides differentiation operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as gradientOps from '../common/ops/gradient';
import { NDArrayCore } from '../common/ndarray-core';
import * as shapeOps from '../common/ops/shape';
import { asarray, full } from './creation';
import { type ArrayLike, fromStorage, fromStorageArray, toStorage } from './types';

/**
 * Expand a `prepend`/`append` value for `diff` to a shape that's compatible
 * with `a` for concatenation along `axis` (NumPy: scalar broadcasts to size 1
 * along axis with `a`'s shape on all other axes).
 */
function expandDiffSide(
  value: ArrayLike,
  a: NDArrayCore,
  axis: number,
): NDArrayCore {
  if (typeof value === 'number' || typeof value === 'bigint') {
    const targetShape = Array.from(a.shape);
    targetShape[axis] = 1;
    return full(targetShape, value, a.dtype as import('../common/dtype').DType);
  }
  return asarray(value);
}

/** Calculate n-th discrete difference */
export function diff(
  a: NDArrayCore,
  n?: number,
  axis?: number,
  prepend?: ArrayLike,
  append?: ArrayLike,
): NDArrayCore {
  if (prepend === undefined && append === undefined) {
    return fromStorage(gradientOps.diff(toStorage(a), n, axis));
  }
  const ndim = toStorage(a).ndim;
  const ax = axis === undefined ? ndim - 1 : axis < 0 ? ndim + axis : axis;
  const parts: NDArrayCore[] = [];
  if (prepend !== undefined) parts.push(expandDiffSide(prepend, a, ax));
  parts.push(a);
  if (append !== undefined) parts.push(expandDiffSide(append, a, ax));
  const combined = shapeOps.concatenate(
    parts.map((p) => toStorage(p)),
    ax,
  );
  return fromStorage(gradientOps.diff(combined, n, axis));
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
        return Array.from(
          toStorage(v).data as { length: number; [n: number]: number | bigint },
          (x) => Number(x),
        );
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
