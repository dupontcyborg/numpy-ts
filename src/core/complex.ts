/**
 * Complex Number Functions - Tree-shakeable standalone functions
 *
 * This module provides complex number operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as complexOps from '../ops/complex';
import { NDArrayCore, toStorage, fromStorage } from './types';

// Re-export Complex class from core
export { Complex } from '../core/complex';

/** Extract real part of array */
export function real(x: NDArrayCore): NDArrayCore {
  return fromStorage(complexOps.real(toStorage(x)));
}

/** Extract imaginary part of array */
export function imag(x: NDArrayCore): NDArrayCore {
  return fromStorage(complexOps.imag(toStorage(x)));
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
