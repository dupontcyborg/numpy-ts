/**
 * Text I/O module for numpy-ts
 *
 * Provides parsing and serialization for delimited text formats (CSV, TSV, etc.).
 * These functions work with strings and are environment-agnostic; for file
 * system operations, use loadtxt/savetxt from 'numpy-ts/node' instead.
 */

export { fromregex, genfromtxt, type ParseTxtOptions, parseTxt } from './parser';
export { type SerializeTxtOptions, serializeTxt } from './serializer';
