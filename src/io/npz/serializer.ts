/**
 * NPZ file serializer
 *
 * Serializes multiple NDArrays to NPZ format (ZIP archive of .npy files).
 */

import { NDArray } from '../../core/ndarray';
import { serializeNpy } from '../npy/serializer';
import { writeZip, writeZipSync } from '../zip/writer';

/**
 * Input type for arrays - supports:
 * - Array of NDArrays (positional, named arr_0, arr_1, etc.)
 * - Map of names to NDArrays
 * - Object with names as keys
 */
export type NpzArraysInput = NDArray[] | Map<string, NDArray> | Record<string, NDArray>;

/**
 * Options for serializing NPZ files
 */
export interface NpzSerializeOptions {
  /**
   * Whether to compress the NPZ file using DEFLATE.
   * Default: false (matches np.savez behavior; use true for np.savez_compressed behavior)
   */
  compress?: boolean;
}

/**
 * Serialize multiple arrays to NPZ format
 *
 * @param arrays - Arrays to save. Can be:
 *   - An array of NDArrays (named arr_0, arr_1, etc. like np.savez positional args)
 *   - A Map of names to NDArrays
 *   - An object with names as keys (like np.savez keyword args)
 * @param options - Serialization options
 * @returns Promise resolving to NPZ file as Uint8Array
 *
 * @example
 * // Positional arrays (named arr_0, arr_1)
 * await serializeNpz([arr1, arr2])
 *
 * // Named arrays
 * await serializeNpz({ x: arr1, y: arr2 })
 */
export async function serializeNpz(
  arrays: NpzArraysInput,
  options: NpzSerializeOptions = {}
): Promise<Uint8Array> {
  const files = prepareNpzFiles(arrays);
  return writeZip(files, { compress: options.compress ?? false });
}

/**
 * Synchronously serialize multiple arrays to NPZ format (no compression)
 *
 * @param arrays - Arrays to save (same input types as serializeNpz)
 * @returns NPZ file as Uint8Array
 */
export function serializeNpzSync(arrays: NpzArraysInput): Uint8Array {
  const files = prepareNpzFiles(arrays);
  return writeZipSync(files);
}

/**
 * Prepare NPY files for ZIP packaging
 */
function prepareNpzFiles(arrays: NpzArraysInput): Map<string, Uint8Array> {
  const files = new Map<string, Uint8Array>();

  // Handle array input (positional arrays get named arr_0, arr_1, etc.)
  if (Array.isArray(arrays)) {
    for (let i = 0; i < arrays.length; i++) {
      const arr = arrays[i]!;
      const npyData = serializeNpy(arr);
      files.set(`arr_${i}.npy`, npyData);
    }
    return files;
  }

  // Handle both Map and plain object
  const entries = arrays instanceof Map ? arrays.entries() : Object.entries(arrays);

  for (const [name, arr] of entries) {
    // Validate array name
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('Array names must be non-empty strings');
    }

    // Serialize to NPY format
    const npyData = serializeNpy(arr);

    // Add .npy extension
    const fileName = name.endsWith('.npy') ? name : `${name}.npy`;
    files.set(fileName, npyData);
  }

  return files;
}
