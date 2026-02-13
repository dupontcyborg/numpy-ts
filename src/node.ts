/**
 * Node.js-specific entry point for numpy-ts
 *
 * This module provides file system operations for saving and loading NPY/NPZ files,
 * plus all core numpy-ts functionality in a single import.
 *
 * Usage:
 *   import * as np from 'numpy-ts/node';
 *   // Has everything from numpy-ts + file I/O
 *   const arr = await np.loadNpy('data.npy');
 *   await np.saveNpz('output.npz', { x: arr });
 *
 * For browser usage, use the main entry point (no file I/O).
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';

import { NDArray } from './full/ndarray';
import { NDArrayCore } from './common/ndarray-core';
import { parseNpy as parseNpyIO } from './io/npy/parser';
import { serializeNpy } from './io/npy/serializer';
import { parseNpz as parseNpzIO, parseNpzSync as parseNpzSyncIO } from './io/npz/parser';
import type { NpzParseOptions } from './io/npz/parser';
import {
  serializeNpz,
  serializeNpzSync,
  type NpzSerializeOptions,
  type NpzArraysInput,
} from './io/npz/serializer';
import {
  parseTxt as parseTxtIO,
  genfromtxt as genfromtxtIO,
  fromregex as fromregexIO,
} from './io/txt';
import { serializeTxt, type ParseTxtOptions, type SerializeTxtOptions } from './io/txt';
import type { DType } from './common/dtype';

// Helper to upgrade NDArrayCore to NDArray
function upgrade(core: NDArrayCore): NDArray {
  return NDArray.fromStorage(core.storage);
}

// NDArray-typed NpzParseResult for node.ts
interface NpzParseResultNDArray {
  arrays: Map<string, NDArray>;
  skipped: string[];
  errors: Map<string, string>;
}

// Re-export everything from the main module for convenience
export * from './index';

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
  return upgrade(parseNpyIO(buffer));
}

/**
 * Synchronously load an NDArray from a .npy file
 *
 * @param path - Path to the .npy file
 * @returns The loaded NDArray
 */
export function loadNpySync(path: string): NDArray {
  const buffer = readFileSync(path);
  return upgrade(parseNpyIO(buffer));
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
): Promise<NpzParseResultNDArray> {
  const buffer = await readFile(path);
  const result = await parseNpzIO(buffer, options);
  const arrays = new Map<string, NDArray>();
  for (const [name, arr] of result.arrays) {
    arrays.set(name, upgrade(arr));
  }
  return { arrays, skipped: result.skipped, errors: result.errors };
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
export function loadNpzFileSync(
  path: string,
  options: NpzParseOptions = {}
): NpzParseResultNDArray {
  const buffer = readFileSync(path);
  const result = parseNpzSyncIO(buffer, options);
  const arrays = new Map<string, NDArray>();
  for (const [name, arr] of result.arrays) {
    arrays.set(name, upgrade(arr));
  }
  return { arrays, skipped: result.skipped, errors: result.errors };
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
): Promise<NDArray | NpzParseResultNDArray> {
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
export function loadSync(path: string, options: LoadOptions = {}): NDArray | NpzParseResultNDArray {
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

// =============================================================================
// Text I/O Functions
// =============================================================================

/**
 * Options for loadtxt
 */
export interface LoadTxtOptions extends ParseTxtOptions {}

/**
 * Options for savetxt
 */
export interface SaveTxtOptions extends SerializeTxtOptions {}

/**
 * Load data from a text file.
 *
 * Each row in the text file must have the same number of values.
 *
 * @param path - Path to the text file
 * @param options - Load options
 * @returns NDArray with the loaded data
 *
 * @example
 * ```typescript
 * // Load a CSV file
 * const arr = await loadtxt('data.csv', { delimiter: ',' });
 *
 * // Load with specific columns
 * const arr = await loadtxt('data.txt', { usecols: [0, 2] });
 *
 * // Skip header rows
 * const arr = await loadtxt('data.txt', { skiprows: 1 });
 * ```
 */
export async function loadtxt(path: string, options: LoadTxtOptions = {}): Promise<NDArray> {
  const content = await readFile(path, { encoding: (options.encoding ?? 'utf-8') as 'utf-8' });
  return upgrade(parseTxtIO(content, options));
}

/**
 * Synchronously load data from a text file.
 *
 * @param path - Path to the text file
 * @param options - Load options
 * @returns NDArray with the loaded data
 */
export function loadtxtSync(path: string, options: LoadTxtOptions = {}): NDArray {
  const content = readFileSync(path, { encoding: (options.encoding ?? 'utf-8') as 'utf-8' });
  return upgrade(parseTxtIO(content, options));
}

/**
 * Save an array to a text file.
 *
 * @param path - Path to save the text file
 * @param arr - The array to save (must be 1D or 2D)
 * @param options - Save options
 *
 * @example
 * ```typescript
 * // Save as CSV
 * await savetxt('data.csv', arr, { delimiter: ',' });
 *
 * // Save with custom format
 * await savetxt('data.txt', arr, { fmt: '%.2f', delimiter: '\t' });
 *
 * // Save with header
 * await savetxt('data.txt', arr, { header: 'x y z' });
 * ```
 */
export async function savetxt(
  path: string,
  arr: NDArray,
  options: SaveTxtOptions = {}
): Promise<void> {
  const content = serializeTxt(arr, options);
  await writeFile(path, content, 'utf-8');
}

/**
 * Synchronously save an array to a text file.
 *
 * @param path - Path to save the text file
 * @param arr - The array to save (must be 1D or 2D)
 * @param options - Save options
 */
export function savetxtSync(path: string, arr: NDArray, options: SaveTxtOptions = {}): void {
  const content = serializeTxt(arr, options);
  writeFileSync(path, content, 'utf-8');
}

/**
 * Load data from a text file with more flexible handling.
 *
 * Similar to loadtxt but handles missing values more gracefully.
 *
 * @param path - Path to the text file
 * @param options - Load options
 * @returns NDArray with the loaded data
 *
 * @example
 * ```typescript
 * // Load file with missing values
 * const arr = await genfromtxt('data.csv', {
 *   delimiter: ',',
 *   missing_values: ['NA', ''],
 *   filling_values: 0
 * });
 * ```
 */
export async function genfromtxt(path: string, options: LoadTxtOptions = {}): Promise<NDArray> {
  const content = await readFile(path, { encoding: (options.encoding ?? 'utf-8') as 'utf-8' });
  return upgrade(genfromtxtIO(content, options));
}

/**
 * Synchronously load data from a text file with more flexible handling.
 *
 * @param path - Path to the text file
 * @param options - Load options
 * @returns NDArray with the loaded data
 */
export function genfromtxtSync(path: string, options: LoadTxtOptions = {}): NDArray {
  const content = readFileSync(path, { encoding: (options.encoding ?? 'utf-8') as 'utf-8' });
  return upgrade(genfromtxtIO(content, options));
}

/**
 * Load data from a text file using regular expressions.
 *
 * @param path - Path to the text file
 * @param regexp - Regular expression with capture groups for extracting values
 * @param dtype - Data type of the resulting array (default: 'float64')
 * @returns NDArray with the extracted data
 *
 * @example
 * ```typescript
 * // Extract x,y pairs from "Point: x=1.0, y=2.0" format
 * const arr = await fromregex('points.txt', /x=([\d.]+), y=([\d.]+)/);
 * ```
 */
export async function fromregex(
  path: string,
  regexp: RegExp | string,
  dtype: DType = 'float64'
): Promise<NDArray> {
  const content = await readFile(path, { encoding: 'utf-8' });
  return upgrade(fromregexIO(content, regexp, dtype));
}

/**
 * Synchronously load data from a text file using regular expressions.
 *
 * @param path - Path to the text file
 * @param regexp - Regular expression with capture groups for extracting values
 * @param dtype - Data type of the resulting array (default: 'float64')
 * @returns NDArray with the extracted data
 */
export function fromregexSync(
  path: string,
  regexp: RegExp | string,
  dtype: DType = 'float64'
): NDArray {
  const content = readFileSync(path, { encoding: 'utf-8' });
  return upgrade(fromregexIO(content, regexp, dtype));
}
