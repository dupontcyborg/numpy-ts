/**
 * Text I/O module for numpy-ts
 *
 * Provides parsing and serialization for delimited text formats (CSV, TSV, etc.).
 * These functions work with strings and are environment-agnostic.
 *
 * For file system operations, use the Node.js-specific entry point:
 *   import { loadtxt, savetxt } from 'numpy-ts/node';
 */

export { fromregex, genfromtxt, type ParseTxtOptions, parseTxt } from './parser';
export { type SerializeTxtOptions, serializeTxt } from './serializer';
