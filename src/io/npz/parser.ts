/**
 * NPZ file parser
 *
 * NPZ is a ZIP archive containing multiple .npy files.
 */

import { NDArray } from '../../core/ndarray';
import { parseNpy } from '../npy/parser';
import { UnsupportedDTypeError } from '../npy/format';
import { readZip, readZipSync } from '../zip/reader';

/**
 * Options for parsing NPZ files
 */
export interface NpzParseOptions {
  /**
   * If true, skip arrays with unsupported dtypes instead of throwing an error.
   * Skipped arrays will not be included in the result.
   * Default: false
   */
  force?: boolean;
}

/**
 * Result of parsing an NPZ file
 */
export interface NpzParseResult {
  /** Successfully parsed arrays */
  arrays: Map<string, NDArray>;
  /** Names of arrays that were skipped due to unsupported dtypes (only when force=true) */
  skipped: string[];
  /** Error messages for skipped arrays */
  errors: Map<string, string>;
}

/**
 * Parse an NPZ file from bytes
 *
 * @param buffer - The NPZ file contents
 * @param options - Parse options
 * @returns Promise resolving to parsed arrays
 */
export async function parseNpz(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptions = {}
): Promise<NpzParseResult> {
  const force = options.force ?? false;
  const files = await readZip(buffer);
  return parseNpzFromFiles(files, force);
}

/**
 * Synchronously parse an NPZ file (only works if not DEFLATE compressed)
 *
 * @param buffer - The NPZ file contents
 * @param options - Parse options
 * @returns Parsed arrays
 */
export function parseNpzSync(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptions = {}
): NpzParseResult {
  const force = options.force ?? false;
  const files = readZipSync(buffer);
  return parseNpzFromFiles(files, force);
}

/**
 * Parse NPZ from already-extracted files
 */
function parseNpzFromFiles(files: Map<string, Uint8Array>, force: boolean): NpzParseResult {
  const arrays = new Map<string, NDArray>();
  const skipped: string[] = [];
  const errors = new Map<string, string>();

  for (const [fileName, data] of files) {
    // NPZ entries should have .npy extension
    if (!fileName.endsWith('.npy')) {
      continue;
    }

    // Extract array name (remove .npy extension)
    const name = fileName.slice(0, -4);

    try {
      const arr = parseNpy(data);
      arrays.set(name, arr);
    } catch (error) {
      if (error instanceof UnsupportedDTypeError && force) {
        // Skip this array but continue processing others
        skipped.push(name);
        errors.set(name, error.message);
      } else {
        // Re-throw all other errors, or UnsupportedDTypeError if force is false
        throw error;
      }
    }
  }

  return { arrays, skipped, errors };
}

/**
 * Convenience function to get arrays as a plain object
 *
 * @param buffer - The NPZ file contents
 * @param options - Parse options
 * @returns Promise resolving to object with array names as keys
 */
export async function loadNpz(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptions = {}
): Promise<Record<string, NDArray>> {
  const result = await parseNpz(buffer, options);
  return Object.fromEntries(result.arrays);
}

/**
 * Synchronous version of loadNpz
 */
export function loadNpzSync(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptions = {}
): Record<string, NDArray> {
  const result = parseNpzSync(buffer, options);
  return Object.fromEntries(result.arrays);
}
