/**
 * Node.js-specific entry point for numpy-ts
 *
 * This module provides file system operations for saving and loading NPY/NPZ files.
 * For browser usage, use the main entry point and handle file I/O separately.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';

import { NDArray } from './core/ndarray';
import { parseNpy } from './io/npy/parser';
import { serializeNpy } from './io/npy/serializer';
import { parseNpz, type NpzParseOptions, type NpzParseResult } from './io/npz/parser';
import { serializeNpz, type NpzSerializeOptions, type NpzArraysInput } from './io/npz/serializer';

// Re-export everything from the main module
export * from './index';

// Re-export IO parsing/serialization functions
export * from './io';

/**
 * Options for loading NPY/NPZ files
 */
export interface LoadOptions extends NpzParseOptions {
  /**
   * If true, allow loading .npy files.
   * Default: true
   */
  allowNpy?: boolean;
}

/**
 * Options for saving NPZ files
 */
export interface SaveNpzOptions extends NpzSerializeOptions {}

/**
 * Load an NDArray from a .npy file
 *
 * @param path - Path to the .npy file
 * @returns The loaded NDArray
 */
export async function loadNpy(path: string): Promise<NDArray> {
  const buffer = await readFile(path);
  return parseNpy(buffer);
}

/**
 * Synchronously load an NDArray from a .npy file
 *
 * @param path - Path to the .npy file
 * @returns The loaded NDArray
 */
export function loadNpySync(path: string): NDArray {
  const buffer = readFileSync(path);
  return parseNpy(buffer);
}

/**
 * Save an NDArray to a .npy file
 *
 * @param path - Path to save the .npy file
 * @param arr - The NDArray to save
 */
export async function saveNpy(path: string, arr: NDArray): Promise<void> {
  const data = serializeNpy(arr);
  await writeFile(path, data);
}

/**
 * Synchronously save an NDArray to a .npy file
 *
 * @param path - Path to save the .npy file
 * @param arr - The NDArray to save
 */
export function saveNpySync(path: string, arr: NDArray): void {
  const data = serializeNpy(arr);
  writeFileSync(path, data);
}

/**
 * Load arrays from a .npz file
 *
 * @param path - Path to the .npz file
 * @param options - Load options
 * @returns Object with array names as keys
 */
export async function loadNpzFile(
  path: string,
  options: NpzParseOptions = {}
): Promise<NpzParseResult> {
  const buffer = await readFile(path);
  return parseNpz(buffer, options);
}

/**
 * Synchronously load arrays from a .npz file
 *
 * Note: Only works if the NPZ file is not DEFLATE compressed.
 *
 * @param path - Path to the .npz file
 * @param options - Load options
 * @returns Object with array names as keys
 */
export function loadNpzFileSync(path: string, options: NpzParseOptions = {}): NpzParseResult {
  const buffer = readFileSync(path);
  // Note: This will throw if the file is compressed
  const { parseNpzSync } = require('./io/npz/parser');
  return parseNpzSync(buffer, options);
}

/**
 * Save arrays to a .npz file
 *
 * @param path - Path to save the .npz file
 * @param arrays - Arrays to save:
 *   - Array of NDArrays (positional, named arr_0, arr_1, etc.)
 *   - Map of names to NDArrays
 *   - Object with names as keys
 * @param options - Save options
 */
export async function saveNpz(
  path: string,
  arrays: NpzArraysInput,
  options: SaveNpzOptions = {}
): Promise<void> {
  const data = await serializeNpz(arrays, options);
  await writeFile(path, data);
}

/**
 * Synchronously save arrays to a .npz file (no compression)
 *
 * @param path - Path to save the .npz file
 * @param arrays - Arrays to save (same types as saveNpz)
 */
export function saveNpzSync(path: string, arrays: NpzArraysInput): void {
  const { serializeNpzSync } = require('./io/npz/serializer');
  const data = serializeNpzSync(arrays);
  writeFileSync(path, data);
}

/**
 * Load an array or arrays from a .npy or .npz file
 *
 * This is a convenience function that auto-detects the file format based on extension.
 *
 * @param path - Path to the file
 * @param options - Load options
 * @returns NDArray for .npy files, or NpzParseResult for .npz files
 */
export async function load(
  path: string,
  options: LoadOptions = {}
): Promise<NDArray | NpzParseResult> {
  if (path.endsWith('.npy')) {
    if (options.allowNpy === false) {
      throw new Error('Loading .npy files is disabled (allowNpy: false)');
    }
    return loadNpy(path);
  } else if (path.endsWith('.npz')) {
    return loadNpzFile(path, options);
  } else {
    throw new Error(`Unknown file extension. Expected .npy or .npz, got: ${path}`);
  }
}

/**
 * Synchronously load an array or arrays from a .npy or .npz file
 *
 * @param path - Path to the file
 * @param options - Load options
 * @returns NDArray for .npy files, or NpzParseResult for .npz files
 */
export function loadSync(path: string, options: LoadOptions = {}): NDArray | NpzParseResult {
  if (path.endsWith('.npy')) {
    if (options.allowNpy === false) {
      throw new Error('Loading .npy files is disabled (allowNpy: false)');
    }
    return loadNpySync(path);
  } else if (path.endsWith('.npz')) {
    return loadNpzFileSync(path, options);
  } else {
    throw new Error(`Unknown file extension. Expected .npy or .npz, got: ${path}`);
  }
}

/**
 * Save an array to a .npy file
 *
 * @param path - Path to save the file (should end with .npy)
 * @param arr - The NDArray to save
 */
export async function save(path: string, arr: NDArray): Promise<void> {
  if (!path.endsWith('.npy')) {
    throw new Error(`save() is for .npy files. Use saveNpz() for .npz files. Got: ${path}`);
  }
  return saveNpy(path, arr);
}

/**
 * Synchronously save an array to a .npy file
 *
 * @param path - Path to save the file (should end with .npy)
 * @param arr - The NDArray to save
 */
export function saveSync(path: string, arr: NDArray): void {
  if (!path.endsWith('.npy')) {
    throw new Error(`saveSync() is for .npy files. Use saveNpzSync() for .npz files. Got: ${path}`);
  }
  return saveNpySync(path, arr);
}

/**
 * Save multiple arrays to a .npz file (like np.savez)
 *
 * @param path - Path to save the .npz file
 * @param arrays - Arrays to save:
 *   - Array of NDArrays: named arr_0, arr_1, etc. (like np.savez positional args)
 *   - Object/Map with names as keys (like np.savez keyword args)
 *
 * @example
 * // Positional arrays
 * await savez('data.npz', [arr1, arr2])  // saved as arr_0, arr_1
 *
 * // Named arrays
 * await savez('data.npz', { x: arr1, y: arr2 })
 */
export async function savez(path: string, arrays: NpzArraysInput): Promise<void> {
  if (!path.endsWith('.npz')) {
    path = path + '.npz';
  }
  return saveNpz(path, arrays, { compress: false });
}

/**
 * Save multiple arrays to a compressed .npz file (like np.savez_compressed)
 *
 * @param path - Path to save the .npz file
 * @param arrays - Arrays to save (same input types as savez)
 */
export async function savez_compressed(path: string, arrays: NpzArraysInput): Promise<void> {
  if (!path.endsWith('.npz')) {
    path = path + '.npz';
  }
  return saveNpz(path, arrays, { compress: true });
}
