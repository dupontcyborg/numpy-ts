/**
 * Shared types for standalone function modules
 *
 * This module provides type definitions and helper functions used by
 * standalone function wrappers for tree-shaking support.
 */

import { Complex } from '../common/complex';
import { NDArrayCore, rawOf } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';

// Re-export types needed by functions
export type { DType, TypedArray } from '../common/dtype';
export { ArrayStorage, Complex, NDArrayCore };

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
    // Read through the raw target: `a.storage` is a Proxy trap dispatch.
    return rawOf(a).storage;
  }
  // Handle NDArray (full/ndarray.ts) which has storage property but doesn't extend NDArrayCore
  // biome-ignore lint/suspicious/noExplicitAny: required for type coercion
  if (a && typeof a === 'object' && 'storage' in a && (a as any).storage instanceof ArrayStorage) {
    // biome-ignore lint/suspicious/noExplicitAny: required for type coercion
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
  // If original has a base, use that; otherwise use original as the base.
  // `original.base` goes through the Proxy; the raw target does not.
  const base = rawOf(original).base ?? original;
  return NDArrayCore.fromStorage(storage, base);
}

/**
 * Wrap a result that *may* alias its input.
 *
 * Several ops return the input's buffer for some dtypes and a fresh one for
 * others — `real` (view always), `imag` (view for complex, zeros for real),
 * `real_if_close` and `round`/`around` (view only when there is nothing to do).
 * Linking those unconditionally would give an independent array a bogus base;
 * not linking them at all leaves an aliasing result claiming to own its data,
 * so `.base` cannot be used to decide whether a write is safe.
 *
 * Testing the buffer identity gets it right in both directions and stays right
 * if the ops layer changes which dtypes take which path.
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
