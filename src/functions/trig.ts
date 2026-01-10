/**
 * Trigonometric Functions - Tree-shakeable standalone functions
 *
 * This module provides trigonometric and hyperbolic functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as trigOps from '../ops/trig';
import * as hyperbolicOps from '../ops/hyperbolic';
import { NDArrayCore, toStorage, fromStorage } from './types';

// ============================================================
// Basic Trigonometric Functions
// ============================================================

/** Sine of array elements */
export function sin(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.sin(toStorage(x)));
}

/** Cosine of array elements */
export function cos(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.cos(toStorage(x)));
}

/** Tangent of array elements */
export function tan(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.tan(toStorage(x)));
}

// ============================================================
// Inverse Trigonometric Functions
// ============================================================

/** Inverse sine */
export function arcsin(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.arcsin(toStorage(x)));
}

/** Alias for arcsin */
export const asin = arcsin;

/** Inverse cosine */
export function arccos(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.arccos(toStorage(x)));
}

/** Alias for arccos */
export const acos = arccos;

/** Inverse tangent */
export function arctan(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.arctan(toStorage(x)));
}

/** Alias for arctan */
export const atan = arctan;

/** Two-argument inverse tangent */
export function arctan2(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(trigOps.arctan2(toStorage(x1), x2Storage));
}

/** Alias for arctan2 */
export const atan2 = arctan2;

// ============================================================
// Other Trigonometric Functions
// ============================================================

/** Hypotenuse (sqrt(x1^2 + x2^2)) */
export function hypot(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(trigOps.hypot(toStorage(x1), x2Storage));
}

/** Convert radians to degrees */
export function degrees(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.degrees(toStorage(x)));
}

/** Convert degrees to radians */
export function radians(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.radians(toStorage(x)));
}

/** Convert degrees to radians (alias) */
export function deg2rad(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.deg2rad(toStorage(x)));
}

/** Convert radians to degrees (alias) */
export function rad2deg(x: NDArrayCore): NDArrayCore {
  return fromStorage(trigOps.rad2deg(toStorage(x)));
}

// ============================================================
// Hyperbolic Functions
// ============================================================

/** Hyperbolic sine */
export function sinh(x: NDArrayCore): NDArrayCore {
  return fromStorage(hyperbolicOps.sinh(toStorage(x)));
}

/** Hyperbolic cosine */
export function cosh(x: NDArrayCore): NDArrayCore {
  return fromStorage(hyperbolicOps.cosh(toStorage(x)));
}

/** Hyperbolic tangent */
export function tanh(x: NDArrayCore): NDArrayCore {
  return fromStorage(hyperbolicOps.tanh(toStorage(x)));
}

// ============================================================
// Inverse Hyperbolic Functions
// ============================================================

/** Inverse hyperbolic sine */
export function arcsinh(x: NDArrayCore): NDArrayCore {
  return fromStorage(hyperbolicOps.arcsinh(toStorage(x)));
}

/** Alias for arcsinh */
export const asinh = arcsinh;

/** Inverse hyperbolic cosine */
export function arccosh(x: NDArrayCore): NDArrayCore {
  return fromStorage(hyperbolicOps.arccosh(toStorage(x)));
}

/** Alias for arccosh */
export const acosh = arccosh;

/** Inverse hyperbolic tangent */
export function arctanh(x: NDArrayCore): NDArrayCore {
  return fromStorage(hyperbolicOps.arctanh(toStorage(x)));
}

/** Alias for arctanh */
export const atanh = arctanh;
