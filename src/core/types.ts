/**
 * Shared types for standalone function modules
 *
 * This module provides type definitions and helper functions used by
 * standalone function wrappers for tree-shaking support.
 */

import { Complex } from '../common/complex';
import type { TypedArray } from '../common/dtype';
import { NDArrayCore, rawOf } from '../common/ndarray-core';
import { ArrayStorage, notAnNDArray } from '../common/storage';

// Re-export types needed by functions
export type { DType, TypedArray } from '../common/dtype';
export { ArrayStorage, Complex, NDArrayCore };

type ArrayLikeScalar = number | bigint | boolean;
type NestedArray<T> = readonly (T | NestedArray<T>)[];

/**
 * Input type for the few functions that take plain data as well as an NDArray: an NDArray, a
 * scalar, a nested array of any depth, or a typed array. Every other function requires an NDArray.
 */
export type ArrayLike = NDArrayCore | ArrayLikeScalar | NestedArray<ArrayLikeScalar> | TypedArray;

/**
 * Convert input to ArrayStorage
 * Handles both NDArrayCore and NDArray (which has _storage property)
 */
export function toStorage(a: NDArrayCore | ArrayStorage): ArrayStorage {
  if (a instanceof NDArrayCore) {
    // Read through the raw target: `a.storage` is a Proxy trap dispatch.
    return rawOf(a).storage;
  }
  // Handle NDArray (full/ndarray.ts) which has storage property but doesn't extend NDArrayCore
  // biome-ignore lint/suspicious/noExplicitAny: required for type coercion
  if (a && typeof a === 'object' && 'storage' in a && (a as any).storage instanceof ArrayStorage) {
    // biome-ignore lint/suspicious/noExplicitAny: required for type coercion
    return (a as any).storage;
  }
  if (a instanceof ArrayStorage) {
    return a;
  }
  throw notAnNDArray(a);
}

/**
 * Throw the standard TypeError unless the value is an NDArray. For functions that call NDArray
 * members directly instead of going through toStorage.
 */
export function assertNDArray(a: unknown): asserts a is NDArrayCore {
  if (!(a instanceof NDArrayCore)) {
    throw notAnNDArray(a);
  }
}

/**
 * Wrap ArrayStorage result in NDArrayCore
 */
export function fromStorage(storage: ArrayStorage, base?: NDArrayCore): NDArrayCore {
  return NDArrayCore.fromStorage(storage, base);
}

/**
 * Wrap ArrayStorage result in NDArrayCore as a view of the input array
 * Sets the base to track the view relationship
 */
export function fromStorageView(storage: ArrayStorage, original: NDArrayCore): NDArrayCore {
  // If original has a base, use that; otherwise use original as the base.
  // `original.base` goes through the Proxy; the raw target does not.
  const base = rawOf(original).base ?? original;
  return NDArrayCore.fromStorage(storage, base);
}

/**
 * Wraps a result that may alias its input. Ops like `real`, `imag`, `real_if_close`,
 * and `round`/`around` return the input's buffer for some dtypes and a fresh one for
 * others, so `.base` alone can't decide whether a write is safe; comparing buffer
 * identity gets it right in both directions and keeps working if the ops layer
 * changes which dtypes take which path.
 */
export function fromStorageMaybeView(storage: ArrayStorage, original: NDArrayCore): NDArrayCore {
  return storage.data === toStorage(original).data
    ? fromStorageView(storage, original)
    : fromStorage(storage);
}

/**
 * Wrap multiple ArrayStorage results
 */
export function fromStorageArray(storages: ArrayStorage[]): NDArrayCore[] {
  return storages.map((s) => fromStorage(s));
}

/**
 * Wrap multiple ArrayStorage results as views of the original array
 */
export function fromStorageViewArray(
  storages: ArrayStorage[],
  original: NDArrayCore,
): NDArrayCore[] {
  return storages.map((s) => fromStorageView(s, original));
}

/**
 * Wrap a tuple of ArrayStorage results
 */
export function fromStorageTuple(tuple: [ArrayStorage, ArrayStorage]): [NDArrayCore, NDArrayCore] {
  return [fromStorage(tuple[0]), fromStorage(tuple[1])];
}
