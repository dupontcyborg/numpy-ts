/**
 * Rounding Functions - Tree-shakeable standalone functions
 *
 * This module provides rounding operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as roundingOps from '../ops/rounding';
import { NDArrayCore, toStorage, fromStorage } from './types';

/** Round to given decimals */
export function around(a: NDArrayCore, decimals: number = 0): NDArrayCore {
  return fromStorage(roundingOps.around(toStorage(a), decimals));
}

/** Alias for around */
export const round_ = around;

/** Round (same as around) */
export function round(a: NDArrayCore, decimals: number = 0): NDArrayCore {
  return fromStorage(roundingOps.round(toStorage(a), decimals));
}

/** Ceiling */
export function ceil(x: NDArrayCore): NDArrayCore {
  return fromStorage(roundingOps.ceil(toStorage(x)));
}

/** Round toward zero */
export function fix(x: NDArrayCore): NDArrayCore {
  return fromStorage(roundingOps.fix(toStorage(x)));
}

/** Floor */
export function floor(x: NDArrayCore): NDArrayCore {
  return fromStorage(roundingOps.floor(toStorage(x)));
}

/** Round to nearest integer */
export function rint(x: NDArrayCore): NDArrayCore {
  return fromStorage(roundingOps.rint(toStorage(x)));
}

/** Truncate toward zero */
export function trunc(x: NDArrayCore): NDArrayCore {
  return fromStorage(roundingOps.trunc(toStorage(x)));
}
