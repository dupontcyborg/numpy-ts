/**
 * IO module for numpy-ts
 *
 * This module provides parsing and serialization for NPY, NPZ, and text formats.
 * These functions work with bytes/strings and are environment-agnostic.
 *
 * For file system operations (save/load), use the Node.js-specific entry point:
 *   import { save, load, loadtxt, savetxt } from 'numpy-ts/node';
 *
 * For browser usage, use fetch or FileReader to get the bytes/text, then use these functions.
 */

export {
  DTYPE_TO_DESCR,
  InvalidNpyError,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
  SUPPORTED_DTYPES,
  UnsupportedDTypeError,
} from './npy/format';
// NPY format
export { parseNpy, parseNpyData, parseNpyHeader } from './npy/parser';
export { serializeNpy } from './npy/serializer';

// NPZ format
export {
  loadNpz,
  loadNpzSync,
  type NpzParseOptions,
  type NpzParseResult,
  parseNpz,
  parseNpzSync,
} from './npz/parser';
export { type NpzSerializeOptions, serializeNpz, serializeNpzSync } from './npz/serializer';

// Text format (CSV, TSV, etc.)
export {
  fromregex,
  genfromtxt,
  type ParseTxtOptions,
  parseTxt,
  type SerializeTxtOptions,
  serializeTxt,
} from './txt';
