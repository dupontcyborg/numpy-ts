/**
 * IO module for numpy-ts
 *
 * This module provides parsing and serialization for NPY and NPZ formats.
 * These functions work with bytes (Uint8Array/ArrayBuffer) and are environment-agnostic.
 *
 * For file system operations (save/load), use the Node.js-specific entry point:
 *   import { save, load } from 'numpy-ts/node';
 *
 * For browser usage, use fetch or FileReader to get the bytes, then use these functions.
 */

// NPY format
export { parseNpy, parseNpyHeader, parseNpyData } from './npy/parser';
export { serializeNpy } from './npy/serializer';
export {
  UnsupportedDTypeError,
  InvalidNpyError,
  SUPPORTED_DTYPES,
  DTYPE_TO_DESCR,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
} from './npy/format';

// NPZ format
export {
  parseNpz,
  parseNpzSync,
  loadNpz,
  loadNpzSync,
  type NpzParseOptions,
  type NpzParseResult,
} from './npz/parser';
export { serializeNpz, serializeNpzSync, type NpzSerializeOptions } from './npz/serializer';
