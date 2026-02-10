/**
 * Type checking and dtype utility functions
 *
 * Tree-shakeable standalone functions for dtype operations.
 */

import { NDArrayCore, type DType } from '../core/ndarray-core';
import { isComplexDType, isFloatDType, isIntegerDType } from '../core/dtype';

// Dtype categories for type checking
const SIGNED_INTEGER_DTYPES: DType[] = ['int8', 'int16', 'int32', 'int64'];
const UNSIGNED_INTEGER_DTYPES: DType[] = ['uint8', 'uint16', 'uint32', 'uint64'];
const INTEGER_DTYPES: DType[] = [...SIGNED_INTEGER_DTYPES, ...UNSIGNED_INTEGER_DTYPES];
const FLOAT_DTYPES: DType[] = ['float32', 'float64'];
const COMPLEX_DTYPES: DType[] = ['complex64', 'complex128'];
const NUMERIC_DTYPES: DType[] = [...INTEGER_DTYPES, ...FLOAT_DTYPES, ...COMPLEX_DTYPES];

// Type hierarchy for casting
const TYPE_HIERARCHY: Record<DType, number> = {
  bool: 0,
  int8: 1,
  uint8: 2,
  int16: 3,
  uint16: 4,
  int32: 5,
  uint32: 6,
  int64: 7,
  uint64: 8,
  float32: 9,
  float64: 10,
  complex64: 11,
  complex128: 12,
};

/**
 * Check if a cast between dtypes is possible according to the casting rule
 */
export function can_cast(
  from_: DType | NDArrayCore,
  to: DType,
  casting: 'no' | 'equiv' | 'safe' | 'same_kind' | 'unsafe' = 'safe'
): boolean {
  const fromDtype: DType = from_ instanceof NDArrayCore ? (from_.dtype as DType) : from_;

  if (casting === 'no') {
    return fromDtype === to;
  }

  if (casting === 'equiv') {
    return fromDtype === to;
  }

  if (casting === 'unsafe') {
    return true;
  }

  const fromRank = TYPE_HIERARCHY[fromDtype] ?? -1;
  const toRank = TYPE_HIERARCHY[to] ?? -1;

  if (casting === 'safe') {
    // Safe casting: no loss of precision
    if (fromRank <= toRank) return true;

    // Special cases
    if (isIntegerDType(fromDtype) && isFloatDType(to)) return true;
    if (isFloatDType(fromDtype) && isComplexDType(to)) return true;

    return false;
  }

  if (casting === 'same_kind') {
    // Same kind: integers to integers, floats to floats, etc.
    if (isIntegerDType(fromDtype) && isIntegerDType(to)) return true;
    if (isFloatDType(fromDtype) && isFloatDType(to)) return true;
    if (isComplexDType(fromDtype) && isComplexDType(to)) return true;
    if (fromDtype === 'bool' && isIntegerDType(to)) return true;

    // Also allow safe promotions
    return can_cast(fromDtype, to, 'safe');
  }

  return false;
}

/**
 * Return a scalar type common to input arrays
 */
export function common_type(...arrays: NDArrayCore[]): DType {
  if (arrays.length === 0) {
    return 'float64';
  }

  let hasComplex = false;
  let maxFloat = false;

  for (const arr of arrays) {
    const dtype = arr.dtype as DType;
    if (isComplexDType(dtype)) {
      hasComplex = true;
      if (dtype === 'complex128') {
        return 'complex128';
      }
    } else if (isFloatDType(dtype)) {
      if (dtype === 'float64') {
        maxFloat = true;
      }
    }
  }

  if (hasComplex) {
    return maxFloat ? 'complex128' : 'complex64';
  }

  return maxFloat ? 'float64' : 'float32';
}

/**
 * Return the type that results from applying promotion rules
 */
export function result_type(...arrays_and_dtypes: (NDArrayCore | DType)[]): DType {
  if (arrays_and_dtypes.length === 0) {
    return 'float64';
  }

  let maxRank = -1;
  let resultDtype: DType = 'float64';

  for (const item of arrays_and_dtypes) {
    const dtype: DType = item instanceof NDArrayCore ? (item.dtype as DType) : item;
    const rank = TYPE_HIERARCHY[dtype] ?? -1;

    if (rank > maxRank) {
      maxRank = rank;
      resultDtype = dtype;
    }
  }

  return resultDtype;
}

/**
 * Return the minimum scalar type needed to represent a value
 */
export function min_scalar_type(val: number | bigint | boolean): DType {
  if (typeof val === 'boolean') {
    return 'bool';
  }

  if (typeof val === 'bigint') {
    if (val >= 0n) {
      if (val <= 255n) return 'uint8';
      if (val <= 65535n) return 'uint16';
      if (val <= 4294967295n) return 'uint32';
      return 'uint64';
    } else {
      if (val >= -128n && val <= 127n) return 'int8';
      if (val >= -32768n && val <= 32767n) return 'int16';
      if (val >= -2147483648n && val <= 2147483647n) return 'int32';
      return 'int64';
    }
  }

  // Number
  if (Number.isInteger(val)) {
    if (val >= 0) {
      if (val <= 255) return 'uint8';
      if (val <= 65535) return 'uint16';
      if (val <= 4294967295) return 'uint32';
      return 'int64';
    } else {
      if (val >= -128 && val <= 127) return 'int8';
      if (val >= -32768 && val <= 32767) return 'int16';
      if (val >= -2147483648 && val <= 2147483647) return 'int32';
      return 'int64';
    }
  }

  // Float
  return 'float64';
}

/**
 * Check if dtype1 is a subtype of dtype2
 */
export function issubdtype(dtype1: DType | NDArrayCore, dtype2: DType | string): boolean {
  const dt1: DType = dtype1 instanceof NDArrayCore ? (dtype1.dtype as DType) : dtype1;

  // Handle string category names
  if (typeof dtype2 === 'string') {
    switch (dtype2) {
      case 'number':
      case 'numeric':
        return NUMERIC_DTYPES.includes(dt1);
      case 'integer':
      case 'int':
        return INTEGER_DTYPES.includes(dt1);
      case 'signedinteger':
        return SIGNED_INTEGER_DTYPES.includes(dt1);
      case 'unsignedinteger':
        return UNSIGNED_INTEGER_DTYPES.includes(dt1);
      case 'floating':
      case 'float':
        return FLOAT_DTYPES.includes(dt1);
      case 'complexfloating':
      case 'complex':
        return COMPLEX_DTYPES.includes(dt1);
      default:
        return dt1 === dtype2;
    }
  }

  return dt1 === dtype2;
}

/**
 * Return a description for the given dtype
 */
export function typename(dtype: DType): string {
  const names: Record<DType, string> = {
    bool: 'bool',
    int8: 'signed char',
    int16: 'short',
    int32: 'int',
    int64: 'long long',
    uint8: 'unsigned char',
    uint16: 'unsigned short',
    uint32: 'unsigned int',
    uint64: 'unsigned long long',
    float32: 'float',
    float64: 'double',
    complex64: 'complex64',
    complex128: 'complex128',
  };

  return names[dtype] || dtype;
}

/**
 * Return the character code of the minimum-size type needed
 */
export function mintypecode(
  typechars: string,
  typeset: string = 'GDFgdf',
  default_: string = 'd'
): string {
  const typeCodeToRank: Record<string, number> = {
    b: 0, // int8
    B: 1, // uint8
    h: 2, // int16
    H: 3, // uint16
    i: 4, // int32
    I: 5, // uint32
    l: 6, // int64
    L: 7, // uint64
    f: 8, // float32
    d: 9, // float64
    F: 10, // complex64
    D: 11, // complex128
    g: 9, // long double (map to float64)
    G: 11, // complex long double (map to complex128)
  };

  let maxRank = -1;
  let result = default_;

  for (const char of typechars) {
    if (typeset.includes(char)) {
      const rank = typeCodeToRank[char] ?? -1;
      if (rank > maxRank) {
        maxRank = rank;
        result = char;
      }
    }
  }

  return result;
}
