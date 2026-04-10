/**
 * All 14 dtypes to test for every public API function.
 *
 * Every function is tested with every dtype — even when it doesn't make
 * semantic sense (e.g., sin(bool)) — to verify that numpy-ts either fails
 * or promotes dtypes the same way NumPy does.
 */
export const ALL_DTYPES = [
  'float64',
  'float32',
  'float16',
  'complex128',
  'complex64',
  'int64',
  'uint64',
  'int32',
  'uint32',
  'int16',
  'uint16',
  'int8',
  'uint8',
  'bool',
] as const;
