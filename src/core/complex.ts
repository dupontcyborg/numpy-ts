/**
 * Complex Number Functions - Tree-shakeable standalone functions
 *
 * This module provides complex number operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as complexOps from '../common/ops/complex';
import { fromStorage, fromStorageMaybeView, type NDArrayCore, toStorage } from './types';

// Re-export Complex class from core
export { Complex } from '../common/complex';

/** Extract real part of array (a view of `x`, as in NumPy) */
export function real(x: NDArrayCore): NDArrayCore {
  return fromStorageMaybeView(complexOps.real(toStorage(x)), x);
}

/** Extract imaginary part of array (a view for complex `x`, zeros otherwise) */
export function imag(x: NDArrayCore): NDArrayCore {
  return fromStorageMaybeView(complexOps.imag(toStorage(x)), x);
}

/** Complex conjugate */
export function conj(x: NDArrayCore): NDArrayCore {
  return fromStorage(complexOps.conj(toStorage(x)));
}

/** Alias for conj */
export const conjugate = conj;

/** Phase angle */
export function angle(x: NDArrayCore, deg?: boolean): NDArrayCore {
  return fromStorage(complexOps.angle(toStorage(x), deg));
}
