/**
 * Shared types for standalone function modules
 *
 * This module provides type definitions and helper functions used by
 * standalone function wrappers for tree-shaking support.
 */

import { NDArrayCore } from '../core/ndarray-core';
import { ArrayStorage } from '../core/storage';
import { Complex } from '../core/complex';

// Re-export types needed by functions
export type { DType, TypedArray } from '../core/dtype';
export { NDArrayCore, ArrayStorage, Complex };

/**
 * Input type for functions that accept arrays
 * Can be an NDArrayCore instance or nested arrays
 */
export type ArrayLike = NDArrayCore | number[] | number[][] | number[][][] | number | bigint;

/**
 * Convert input to ArrayStorage
 */
export function toStorage(a: NDArrayCore | ArrayStorage): ArrayStorage {
  if (a instanceof NDArrayCore) {
    return a.storage;
  }
  return a;
}

/**
 * Wrap ArrayStorage result in NDArrayCore
 */
export function fromStorage(storage: ArrayStorage): NDArrayCore {
  return NDArrayCore._fromStorage(storage);
}

/**
 * Wrap multiple ArrayStorage results
 */
export function fromStorageArray(storages: ArrayStorage[]): NDArrayCore[] {
  return storages.map(fromStorage);
}

/**
 * Wrap a tuple of ArrayStorage results
 */
export function fromStorageTuple(tuple: [ArrayStorage, ArrayStorage]): [NDArrayCore, NDArrayCore] {
  return [fromStorage(tuple[0]), fromStorage(tuple[1])];
}
