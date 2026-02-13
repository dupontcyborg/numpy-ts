/**
 * Shape Manipulation Functions - Tree-shakeable standalone functions
 *
 * This module provides array shape manipulation functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as shapeOps from '../common/ops/shape';
import {
  NDArrayCore,
  toStorage,
  fromStorage,
  fromStorageView,
  fromStorageArray,
  fromStorageViewArray,
} from './types';

// ============================================================
// Basic Shape Manipulation
// ============================================================

/** Reshape array without changing data */
export function reshape(a: NDArrayCore, newShape: number[]): NDArrayCore {
  return fromStorage(shapeOps.reshape(toStorage(a), newShape));
}

/** Flatten array to 1D */
export function flatten(a: NDArrayCore): NDArrayCore {
  return fromStorage(shapeOps.flatten(toStorage(a)));
}

/** Return contiguous flattened array */
export function ravel(a: NDArrayCore): NDArrayCore {
  return fromStorage(shapeOps.ravel(toStorage(a)));
}

/** Remove single-dimensional entries from shape - returns a view */
export function squeeze(a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorageView(shapeOps.squeeze(toStorage(a), axis), a);
}

/** Expand array dimensions - returns a view */
export function expand_dims(a: NDArrayCore, axis: number): NDArrayCore {
  return fromStorageView(shapeOps.expandDims(toStorage(a), axis), a);
}

// ============================================================
// Axis Manipulation
// ============================================================

/** Interchange two axes - returns a view */
export function swapaxes(a: NDArrayCore, axis1: number, axis2: number): NDArrayCore {
  return fromStorageView(shapeOps.swapaxes(toStorage(a), axis1, axis2), a);
}

/** Move axis to new position - returns a view */
export function moveaxis(
  a: NDArrayCore,
  source: number | number[],
  destination: number | number[]
): NDArrayCore {
  return fromStorageView(shapeOps.moveaxis(toStorage(a), source, destination), a);
}

/** Roll axis to given position - returns a view */
export function rollaxis(a: NDArrayCore, axis: number, start: number = 0): NDArrayCore {
  return fromStorageView(shapeOps.rollaxis(toStorage(a), axis, start), a);
}

// ============================================================
// Joining Arrays
// ============================================================

/** Join arrays along an existing axis */
export function concatenate(arrays: NDArrayCore[], axis: number = 0): NDArrayCore {
  return fromStorage(
    shapeOps.concatenate(
      arrays.map((a) => toStorage(a)),
      axis
    )
  );
}

/** Join arrays along a new axis */
export function stack(arrays: NDArrayCore[], axis: number = 0): NDArrayCore {
  return fromStorage(
    shapeOps.stack(
      arrays.map((a) => toStorage(a)),
      axis
    )
  );
}

/** Stack arrays vertically (row-wise) */
export function vstack(arrays: NDArrayCore[]): NDArrayCore {
  return fromStorage(shapeOps.vstack(arrays.map((a) => toStorage(a))));
}

/** Stack arrays horizontally (column-wise) */
export function hstack(arrays: NDArrayCore[]): NDArrayCore {
  return fromStorage(shapeOps.hstack(arrays.map((a) => toStorage(a))));
}

/** Stack arrays along third axis (depth-wise) */
export function dstack(arrays: NDArrayCore[]): NDArrayCore {
  return fromStorage(shapeOps.dstack(arrays.map((a) => toStorage(a))));
}

/** Concatenate (alias) */
export function concat(arrays: NDArrayCore[], axis: number = 0): NDArrayCore {
  return fromStorage(
    shapeOps.concat(
      arrays.map((a) => toStorage(a)),
      axis
    )
  );
}

/** Stack 1D arrays as columns */
export function column_stack(arrays: NDArrayCore[]): NDArrayCore {
  return fromStorage(shapeOps.columnStack(arrays.map((a) => toStorage(a))));
}

/** vstack alias */
export const row_stack = vstack;

/** Assemble arrays from nested sequences of blocks */
export function block(arrays: NDArrayCore[]): NDArrayCore {
  return fromStorage(shapeOps.block(arrays.map((a) => toStorage(a))));
}

// ============================================================
// Splitting Arrays
// ============================================================

/** Split array into multiple sub-arrays - returns views */
export function split(
  a: NDArrayCore,
  indicesOrSections: number | number[],
  axis: number = 0
): NDArrayCore[] {
  return fromStorageViewArray(shapeOps.split(toStorage(a), indicesOrSections, axis), a);
}

/** Split array into multiple sub-arrays (may have unequal sizes) - returns views */
export function array_split(
  a: NDArrayCore,
  indicesOrSections: number | number[],
  axis: number = 0
): NDArrayCore[] {
  return fromStorageViewArray(shapeOps.arraySplit(toStorage(a), indicesOrSections, axis), a);
}

/** Split array vertically - returns views */
export function vsplit(a: NDArrayCore, indicesOrSections: number | number[]): NDArrayCore[] {
  return fromStorageViewArray(shapeOps.vsplit(toStorage(a), indicesOrSections), a);
}

/** Split array horizontally - returns views */
export function hsplit(a: NDArrayCore, indicesOrSections: number | number[]): NDArrayCore[] {
  return fromStorageViewArray(shapeOps.hsplit(toStorage(a), indicesOrSections), a);
}

/** Split array along third axis */
/** Split array along depth axis - returns views */
export function dsplit(ary: NDArrayCore, indices_or_sections: number | number[]): NDArrayCore[] {
  return fromStorageViewArray(shapeOps.dsplit(toStorage(ary), indices_or_sections), ary);
}

/** Unstack array into list of arrays */
export function unstack(a: NDArrayCore, axis: number = 0): NDArrayCore[] {
  return fromStorageArray(shapeOps.unstack(toStorage(a), axis));
}

// ============================================================
// Tiling and Repeating
// ============================================================

/** Tile array by given repetitions */
export function tile(a: NDArrayCore, reps: number | number[]): NDArrayCore {
  return fromStorage(shapeOps.tile(toStorage(a), reps));
}

/** Repeat elements of array */
export function repeat(a: NDArrayCore, repeats: number | number[], axis?: number): NDArrayCore {
  return fromStorage(shapeOps.repeat(toStorage(a), repeats, axis));
}

// ============================================================
// Flipping and Rotating
// ============================================================

/** Reverse order of elements along axis */
export function flip(m: NDArrayCore, axis?: number | number[]): NDArrayCore {
  return fromStorage(shapeOps.flip(toStorage(m), axis));
}

/** Flip array left/right */
export function fliplr(m: NDArrayCore): NDArrayCore {
  return flip(m, 1);
}

/** Flip array up/down */
export function flipud(m: NDArrayCore): NDArrayCore {
  return flip(m, 0);
}

/** Rotate array by 90 degrees */
export function rot90(m: NDArrayCore, k: number = 1, axes: [number, number] = [0, 1]): NDArrayCore {
  return fromStorage(shapeOps.rot90(toStorage(m), k, axes));
}

/** Roll array elements along axis */
export function roll(
  a: NDArrayCore,
  shift: number | number[],
  axis?: number | number[]
): NDArrayCore {
  return fromStorage(shapeOps.roll(toStorage(a), shift, axis));
}

// ============================================================
// Resizing
// ============================================================

/** Resize array to new shape */
export function resize(a: NDArrayCore, new_shape: number[]): NDArrayCore {
  return fromStorage(shapeOps.resize(toStorage(a), new_shape));
}

// ============================================================
// Dimension Guarantees
// ============================================================

/** Ensure array has at least 1 dimension */
export function atleast_1d(...arrays: NDArrayCore[]): NDArrayCore | NDArrayCore[] {
  const result = shapeOps.atleast1d(arrays.map((a) => toStorage(a)));
  if (result.length === 1) {
    return fromStorage(result[0]!);
  }
  return fromStorageArray(result);
}

/** Ensure array has at least 2 dimensions */
export function atleast_2d(...arrays: NDArrayCore[]): NDArrayCore | NDArrayCore[] {
  const result = shapeOps.atleast2d(arrays.map((a) => toStorage(a)));
  if (result.length === 1) {
    return fromStorage(result[0]!);
  }
  return fromStorageArray(result);
}

/** Ensure array has at least 3 dimensions */
export function atleast_3d(...arrays: NDArrayCore[]): NDArrayCore | NDArrayCore[] {
  const result = shapeOps.atleast3d(arrays.map((a) => toStorage(a)));
  if (result.length === 1) {
    return fromStorage(result[0]!);
  }
  return fromStorageArray(result);
}
