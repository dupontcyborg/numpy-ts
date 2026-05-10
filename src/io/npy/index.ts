/**
 * NPY format reading and writing
 */

export {
  DTYPE_TO_DESCR,
  type DTypeParseResult,
  InvalidNpyError,
  isSystemLittleEndian,
  NPY_MAGIC,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
  parseDescriptor,
  SUPPORTED_DTYPES,
  UnsupportedDTypeError,
} from './format';
export { parseNpy, parseNpyData, parseNpyHeader } from './parser';
export { serializeNpy } from './serializer';
