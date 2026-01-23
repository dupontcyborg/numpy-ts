/**
 * Bitwise Functions - Tree-shakeable standalone functions
 *
 * This module provides bitwise operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as bitwiseOps from '../ops/bitwise';
import { NDArrayCore, toStorage, fromStorage } from './types';

// ============================================================
// Bitwise Operations
// ============================================================

/** Bitwise AND */
export function bitwise_and(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.bitwise_and(toStorage(x1), x2Storage));
}

/** Bitwise OR */
export function bitwise_or(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.bitwise_or(toStorage(x1), x2Storage));
}

/** Bitwise XOR */
export function bitwise_xor(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.bitwise_xor(toStorage(x1), x2Storage));
}

/** Bitwise NOT */
export function bitwise_not(x: NDArrayCore): NDArrayCore {
  return fromStorage(bitwiseOps.bitwise_not(toStorage(x)));
}

/** Bitwise inversion (alias for bitwise_not) */
export function invert(x: NDArrayCore): NDArrayCore {
  return fromStorage(bitwiseOps.invert(toStorage(x)));
}

/** Left shift */
export function left_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.left_shift(toStorage(x1), x2Storage));
}

/** Right shift */
export function right_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.right_shift(toStorage(x1), x2Storage));
}

// ============================================================
// Bit Packing
// ============================================================

/** Pack bits into bytes */
export function packbits(a: NDArrayCore, axis?: number, bitorder?: 'big' | 'little'): NDArrayCore {
  return fromStorage(bitwiseOps.packbits(toStorage(a), axis, bitorder));
}

/** Unpack bytes into bits */
export function unpackbits(
  a: NDArrayCore,
  axis?: number,
  count?: number,
  bitorder?: 'big' | 'little'
): NDArrayCore {
  return fromStorage(bitwiseOps.unpackbits(toStorage(a), axis, count, bitorder));
}

// ============================================================
// Bit Counting
// ============================================================

/** Count number of 1 bits */
export function bitwise_count(x: NDArrayCore): NDArrayCore {
  return fromStorage(bitwiseOps.bitwise_count(toStorage(x)));
}

// ============================================================
// Aliases (Array API standard)
// ============================================================

/** Alias for bitwise_not */
export function bitwise_invert(x: NDArrayCore): NDArrayCore {
  return fromStorage(bitwiseOps.bitwise_invert(toStorage(x)));
}

/** Alias for left_shift */
export function bitwise_left_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.bitwise_left_shift(toStorage(x1), x2Storage));
}

/** Alias for right_shift */
export function bitwise_right_shift(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(bitwiseOps.bitwise_right_shift(toStorage(x1), x2Storage));
}
