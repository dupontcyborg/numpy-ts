/**
 * Shared types for standalone function modules
 *
 * This module provides type definitions and helper functions used by
 * standalone function wrappers for tree-shaking support.
 */

import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import { Complex } from '../common/complex';

// Re-export types needed by functions
export type { DType, TypedArray } from '../common/dtype';
export { NDArrayCore, ArrayStorage, Complex };

/**
 * Input type for functions that accept arrays
 * Can be an NDArrayCore instance or nested arrays
 */
export type ArrayLike = NDArrayCore | number[] | number[][] | number[][][] | number | bigint;

/**
 * Convert input to ArrayStorage
 * Handles both NDArrayCore and NDArray (which has _storage property)
 */
export function toStorage(a: NDArrayCore | ArrayStorage): ArrayStorage {
  if (a instanceof NDArrayCore) {
    return a.storage;
  }
  // Handle NDArray (full/ndarray.ts) which has storage property but doesn't extend NDArrayCore
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (a && typeof a === 'object' && 'storage' in a && (a as any).storage instanceof ArrayStorage) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (a as any).storage;
  }
  return a;
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
  // If original has a base, use that; otherwise use original as the base
  const base = original.base ?? original;
  return NDArrayCore.fromStorage(storage, base);
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
  original: NDArrayCore
): NDArrayCore[] {
  return storages.map((s) => fromStorageView(s, original));
}

/**
 * Wrap a tuple of ArrayStorage results
 */
export function fromStorageTuple(tuple: [ArrayStorage, ArrayStorage]): [NDArrayCore, NDArrayCore] {
  return [fromStorage(tuple[0]), fromStorage(tuple[1])];
}
