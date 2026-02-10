/**
 * Advanced indexing and data manipulation functions
 *
 * Tree-shakeable standalone functions that wrap the underlying ops.
 */

import { NDArrayCore } from '../core/ndarray-core';
import { ArrayStorage } from '../core/storage';
import * as advancedOps from '../ops/advanced';
import * as comparisonOps from '../ops/comparison';
import { array } from '../creation';

// Helper to convert NDArrayCore to ArrayStorage
function toStorage(a: NDArrayCore): ArrayStorage {
  return (a as unknown as { _storage: ArrayStorage })._storage;
}

// Helper to convert ArrayStorage to NDArrayCore
function fromStorage(storage: ArrayStorage): NDArrayCore {
  return new NDArrayCore(storage);
}

// ============================================================
// Broadcasting
// ============================================================

export function broadcast_to(a: NDArrayCore, shape: number[]): NDArrayCore {
  return fromStorage(advancedOps.broadcast_to(toStorage(a), shape));
}

export function broadcast_arrays(...arrays: NDArrayCore[]): NDArrayCore[] {
  const storages = arrays.map(toStorage);
  return advancedOps.broadcast_arrays(storages).map(fromStorage);
}

export function broadcast_shapes(...shapes: number[][]): number[] {
  if (shapes.length === 0) return [];
  if (shapes.length === 1) return [...shapes[0]!];

  let result = [...shapes[0]!];
  for (let i = 1; i < shapes.length; i++) {
    const shape = shapes[i]!;
    const maxLen = Math.max(result.length, shape.length);
    const newResult: number[] = [];

    for (let j = 0; j < maxLen; j++) {
      const dim1 = j < result.length ? result[result.length - 1 - j]! : 1;
      const dim2 = j < shape.length ? shape[shape.length - 1 - j]! : 1;

      if (dim1 === dim2) {
        newResult.unshift(dim1);
      } else if (dim1 === 1) {
        newResult.unshift(dim2);
      } else if (dim2 === 1) {
        newResult.unshift(dim1);
      } else {
        throw new Error(`Cannot broadcast shapes: dimensions ${dim1} and ${dim2} are incompatible`);
      }
    }
    result = newResult;
  }

  return result;
}

// ============================================================
// Indexing Operations
// ============================================================

export function take(a: NDArrayCore, indices: number[], axis?: number): NDArrayCore {
  return fromStorage(advancedOps.take(toStorage(a), indices, axis));
}

export function put(a: NDArrayCore, indices: number[], values: NDArrayCore | number[]): void {
  const valuesStorage = Array.isArray(values) ? toStorage(array(values)) : toStorage(values);
  advancedOps.put(toStorage(a), indices, valuesStorage);
}

export function take_along_axis(arr: NDArrayCore, indices: NDArrayCore, axis: number): NDArrayCore {
  return fromStorage(advancedOps.take_along_axis(toStorage(arr), toStorage(indices), axis));
}

export function put_along_axis(
  arr: NDArrayCore,
  indices: NDArrayCore,
  values: NDArrayCore,
  axis: number
): void {
  advancedOps.put_along_axis(toStorage(arr), toStorage(indices), toStorage(values), axis);
}

export function choose(a: NDArrayCore, choices: NDArrayCore[]): NDArrayCore {
  const choiceStorages = choices.map(toStorage);
  return fromStorage(advancedOps.choose(toStorage(a), choiceStorages));
}

export function compress(condition: NDArrayCore, a: NDArrayCore, axis?: number): NDArrayCore {
  return fromStorage(advancedOps.compress(toStorage(condition), toStorage(a), axis));
}

/**
 * Integer array indexing (fancy indexing)
 *
 * Select elements using an array of indices.
 */
export function iindex(
  a: NDArrayCore,
  indices: NDArrayCore | number[] | number[][],
  axis: number = 0
): NDArrayCore {
  // Convert NDArrayCore or nested array to flat number[]
  let indexArray: number[];
  if (indices instanceof NDArrayCore) {
    indexArray = Array.from(indices.data as unknown as number[]);
  } else if (Array.isArray(indices[0])) {
    // Flatten nested array
    indexArray = (indices as number[][]).flat();
  } else {
    indexArray = indices as number[];
  }
  return take(a, indexArray, axis);
}

/**
 * Boolean array indexing (fancy indexing with mask)
 *
 * Select elements where a boolean mask is true.
 */
export function bindex(a: NDArrayCore, mask: NDArrayCore, axis?: number): NDArrayCore {
  return compress(mask, a, axis);
}

export function select(
  condlist: NDArrayCore[],
  choicelist: NDArrayCore[],
  defaultVal: number = 0
): NDArrayCore {
  const condStorages = condlist.map(toStorage);
  const choiceStorages = choicelist.map(toStorage);
  return fromStorage(advancedOps.select(condStorages, choiceStorages, defaultVal));
}

export function place(a: NDArrayCore, mask: NDArrayCore, vals: NDArrayCore): void {
  advancedOps.place(toStorage(a), toStorage(mask), toStorage(vals));
}

export function putmask(a: NDArrayCore, mask: NDArrayCore, values: NDArrayCore): void {
  advancedOps.putmask(toStorage(a), toStorage(mask), toStorage(values));
}

export function copyto(dst: NDArrayCore, src: NDArrayCore): void {
  const dstData = dst.data;
  const srcData = src.data;
  for (let i = 0; i < Math.min(dstData.length, srcData.length); i++) {
    (dstData as unknown as number[])[i] = srcData[i] as number;
  }
}

// ============================================================
// Index Arrays
// ============================================================

export function indices(
  dimensions: number[],
  dtype: 'int32' | 'int64' | 'float64' = 'int32'
): NDArrayCore {
  return fromStorage(advancedOps.indices(dimensions, dtype));
}

export function ix_(...args: NDArrayCore[]): NDArrayCore[] {
  const storages = args.map(toStorage);
  return advancedOps.ix_(...storages).map(fromStorage);
}

export function ravel_multi_index(
  multi_index: NDArrayCore[],
  dims: number[],
  mode: 'raise' | 'wrap' | 'clip' = 'raise'
): NDArrayCore {
  const indexStorages = multi_index.map(toStorage);
  return fromStorage(advancedOps.ravel_multi_index(indexStorages, dims, mode));
}

export function unravel_index(indices: NDArrayCore | number, shape: number[]): NDArrayCore[] {
  const indicesStorage =
    typeof indices === 'number' ? toStorage(array([indices])) : toStorage(indices);
  return advancedOps.unravel_index(indicesStorage, shape).map(fromStorage);
}

// ============================================================
// Diagonal Indices
// ============================================================

export function diag_indices(n: number, ndim: number = 2): NDArrayCore[] {
  return advancedOps.diag_indices(n, ndim).map(fromStorage);
}

export function diag_indices_from(a: NDArrayCore): NDArrayCore[] {
  return advancedOps.diag_indices_from(toStorage(a)).map(fromStorage);
}

export function fill_diagonal(a: NDArrayCore, val: number, wrap: boolean = false): void {
  advancedOps.fill_diagonal(toStorage(a), val, wrap);
}

// ============================================================
// Triangular Indices
// ============================================================

export function tril_indices(n: number, k: number = 0, m?: number): NDArrayCore[] {
  return advancedOps.tril_indices(n, k, m).map(fromStorage);
}

export function tril_indices_from(a: NDArrayCore, k: number = 0): NDArrayCore[] {
  return advancedOps.tril_indices_from(toStorage(a), k).map(fromStorage);
}

export function triu_indices(n: number, k: number = 0, m?: number): NDArrayCore[] {
  return advancedOps.triu_indices(n, k, m).map(fromStorage);
}

export function triu_indices_from(a: NDArrayCore, k: number = 0): NDArrayCore[] {
  return advancedOps.triu_indices_from(toStorage(a), k).map(fromStorage);
}

export function mask_indices(
  n: number,
  mask_func: (n: number, k: number) => NDArrayCore,
  k: number = 0
): NDArrayCore[] {
  const wrappedMaskFunc = (nn: number, kk: number) => toStorage(mask_func(nn, kk));
  return advancedOps.mask_indices(n, wrappedMaskFunc, k).map(fromStorage);
}

// ============================================================
// Array Comparison
// ============================================================

export function array_equal(a: NDArrayCore, b: NDArrayCore, equal_nan: boolean = false): boolean {
  return advancedOps.array_equal(toStorage(a), toStorage(b), equal_nan);
}

export function array_equiv(a: NDArrayCore, b: NDArrayCore): boolean {
  return comparisonOps.arrayEquiv(toStorage(a), toStorage(b));
}

// ============================================================
// Apply Functions
// ============================================================

export function apply_along_axis(
  func1d: (arr: NDArrayCore) => NDArrayCore | number,
  axis: number,
  arr: NDArrayCore
): NDArrayCore {
  const wrappedFunc = (storage: ArrayStorage): ArrayStorage | number => {
    const result = func1d(fromStorage(storage));
    return typeof result === 'number' ? result : toStorage(result);
  };
  return fromStorage(advancedOps.apply_along_axis(toStorage(arr), axis, wrappedFunc));
}

export function apply_over_axes(
  func: (arr: NDArrayCore, axis: number) => NDArrayCore,
  a: NDArrayCore,
  axes: number[]
): NDArrayCore {
  const wrappedFunc = (storage: ArrayStorage, axis: number): ArrayStorage => {
    return toStorage(func(fromStorage(storage), axis));
  };
  return fromStorage(advancedOps.apply_over_axes(toStorage(a), wrappedFunc, axes));
}

// ============================================================
// Memory Functions
// ============================================================

export function may_share_memory(a: NDArrayCore, b: NDArrayCore): boolean {
  return advancedOps.may_share_memory(toStorage(a), toStorage(b));
}

export function shares_memory(a: NDArrayCore, b: NDArrayCore): boolean {
  return advancedOps.shares_memory(toStorage(a), toStorage(b));
}

// ============================================================
// Error State
// ============================================================

export const geterr = advancedOps.geterr;
export const seterr = advancedOps.seterr;
