/**
 * NPY file format constants and type definitions
 *
 * NPY is NumPy's native binary format for storing arrays.
 * Spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 */

import type { DType } from '../../core/dtype';

/**
 * NPY magic number: \x93NUMPY (6 bytes)
 */
export const NPY_MAGIC = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);

/**
 * Supported NPY format versions
 * - v1.0: 2-byte header length (max 65535 bytes)
 * - v2.0: 4-byte header length (max 4GB)
 * - v3.0: allows UTF-8 in description (same as v2 otherwise)
 *
 * We read v1, v2, and v3; we write v2 only
 */
export interface NpyVersion {
  major: number;
  minor: number;
}

/**
 * NPY header information
 */
export interface NpyHeader {
  /** Data type descriptor (e.g., '<f8', '>i4') */
  descr: string;
  /** Whether array is Fortran-contiguous (column-major) */
  fortran_order: boolean;
  /** Array shape */
  shape: number[];
}

/**
 * Parsed NPY metadata including version
 */
export interface NpyMetadata {
  version: NpyVersion;
  header: NpyHeader;
  /** Byte offset where data starts */
  dataOffset: number;
}

/**
 * Result of parsing an NPY header descriptor to our DType
 */
export interface DTypeParseResult {
  dtype: DType;
  /** Whether the data needs byte swapping (big-endian on little-endian or vice versa) */
  needsByteSwap: boolean;
  /** Element size in bytes */
  itemsize: number;
}

/**
 * All dtypes we support
 */
export const SUPPORTED_DTYPES: DType[] = [
  'float64',
  'float32',
  'int64',
  'int32',
  'int16',
  'int8',
  'uint64',
  'uint32',
  'uint16',
  'uint8',
  'bool',
];

/**
 * Detect system endianness
 */
export function isSystemLittleEndian(): boolean {
  const buffer = new ArrayBuffer(2);
  new DataView(buffer).setInt16(0, 256, true);
  return new Int16Array(buffer)[0] === 256;
}

/**
 * NPY descriptor to DType mapping
 *
 * NumPy descriptors follow the format: <endian><type><size>
 * - Endian: '<' little, '>' big, '=' native, '|' not applicable (1-byte types)
 * - Type: 'f' float, 'i' signed int, 'u' unsigned int, 'b' bool, 'c' complex, etc.
 * - Size: byte size (1, 2, 4, 8)
 */
const DESCR_TO_DTYPE: Record<string, DType> = {
  // Float types
  f8: 'float64',
  f4: 'float32',
  // Signed integer types
  i8: 'int64',
  i4: 'int32',
  i2: 'int16',
  i1: 'int8',
  // Unsigned integer types
  u8: 'uint64',
  u4: 'uint32',
  u2: 'uint16',
  u1: 'uint8',
  // Boolean
  b1: 'bool',
};

/**
 * DType to NPY descriptor mapping (for serialization)
 * We always write little-endian
 */
export const DTYPE_TO_DESCR: Record<DType, string> = {
  float64: '<f8',
  float32: '<f4',
  int64: '<i8',
  int32: '<i4',
  int16: '<i2',
  int8: '|i1',
  uint64: '<u8',
  uint32: '<u4',
  uint16: '<u2',
  uint8: '|u1',
  bool: '|b1',
};

/**
 * Unsupported dtype types (for error messages)
 */
export const UNSUPPORTED_DTYPE_PATTERNS: Record<string, string> = {
  c: 'complex numbers',
  S: 'byte strings',
  U: 'Unicode strings',
  O: 'Python objects',
  V: 'structured arrays (void)',
  M: 'datetime64',
  m: 'timedelta64',
};

/**
 * Parse a NumPy dtype descriptor string to our DType
 *
 * @param descr - NumPy descriptor like '<f8', '>i4', '|b1'
 * @returns Parsed result with dtype and byte order info
 * @throws Error if dtype is not supported
 */
export function parseDescriptor(descr: string): DTypeParseResult {
  // Handle structured dtypes (tuples/lists) - not supported
  if (descr.startsWith('[') || descr.startsWith('(')) {
    throw new UnsupportedDTypeError(`Structured/compound dtypes are not supported: ${descr}`);
  }

  // Extract endianness, type, and size
  let endian = '';
  let typeAndSize = descr;

  // Check for endian prefix
  if (descr[0] === '<' || descr[0] === '>' || descr[0] === '=' || descr[0] === '|') {
    endian = descr[0];
    typeAndSize = descr.slice(1);
  }

  // Check for unsupported types
  const typeChar = typeAndSize[0];
  if (typeChar && typeChar in UNSUPPORTED_DTYPE_PATTERNS) {
    throw new UnsupportedDTypeError(
      `Unsupported dtype: ${UNSUPPORTED_DTYPE_PATTERNS[typeChar]} (${descr}). ` +
        `Use the 'force' parameter to skip arrays with unsupported dtypes.`
    );
  }

  // Look up in our mapping
  const dtype = DESCR_TO_DTYPE[typeAndSize];
  if (!dtype) {
    throw new UnsupportedDTypeError(
      `Unknown or unsupported dtype descriptor: ${descr}. ` +
        `Supported types: ${SUPPORTED_DTYPES.join(', ')}. ` +
        `Use the 'force' parameter to skip arrays with unsupported dtypes.`
    );
  }

  // Determine if byte swapping is needed
  const isLittleEndian = isSystemLittleEndian();
  const dataIsLittleEndian = endian === '<' || endian === '|' || (endian === '=' && isLittleEndian);
  const dataIsBigEndian = endian === '>' || (endian === '=' && !isLittleEndian);

  // We need to byte swap if:
  // - Data is big-endian and system is little-endian
  // - Data is little-endian and system is big-endian
  // But only for multi-byte types
  const itemsize = parseInt(typeAndSize.slice(1), 10);
  const needsByteSwap =
    itemsize > 1 &&
    ((dataIsBigEndian && isLittleEndian) || (dataIsLittleEndian && !isLittleEndian));

  return {
    dtype,
    needsByteSwap,
    itemsize,
  };
}

/**
 * Custom error for unsupported dtypes
 */
export class UnsupportedDTypeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedDTypeError';
  }
}

/**
 * Custom error for invalid NPY format
 */
export class InvalidNpyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidNpyError';
  }
}
