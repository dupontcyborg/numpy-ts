/**
 * Type checking and dtype utility functions
 *
 * Tree-shakeable standalone functions for dtype operations.
 */

import { NDArrayCore, type DType } from '../common/ndarray-core';
import { isComplexDType, isFloatDType, isIntegerDType } from '../common/dtype';

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

    // Integer to float: only safe if float can represent all values
    // This check must come BEFORE the rank comparison because int32->float32
    // has increasing rank but is NOT safe (precision loss)
    if (isIntegerDType(fromDtype) && isFloatDType(to)) {
      // float32 has 24-bit mantissa, can safely hold int8, int16, uint8, uint16
      // float64 has 53-bit mantissa, can safely hold int8-int32, uint8-uint32
      const intBits: Record<DType, number> = {
        int8: 8,
        int16: 16,
        int32: 32,
        int64: 64,
        uint8: 8,
        uint16: 16,
        uint32: 32,
        uint64: 64,
        bool: 1,
        float32: 0,
        float64: 0,
        complex64: 0,
        complex128: 0,
      };
      const floatMantissaBits: Record<DType, number> = {
        float32: 24,
        float64: 53,
        int8: 0,
        int16: 0,
        int32: 0,
        int64: 0,
        uint8: 0,
        uint16: 0,
        uint32: 0,
        uint64: 0,
        bool: 0,
        complex64: 24,
        complex128: 53,
      };
      const srcBits = intBits[fromDtype] ?? 64;
      const dstBits = floatMantissaBits[to] ?? 0;
      return srcBits <= dstBits;
    }

    // Integer to complex: NumPy allows any integer to complex (conceptually "wider")
    // This differs from int->float which considers precision loss
    if (isIntegerDType(fromDtype) && isComplexDType(to)) {
      return true;
    }

    // Float to complex: always safe
    if (isFloatDType(fromDtype) && isComplexDType(to)) return true;

    // For same-kind types, use rank comparison
    if (fromRank <= toRank) return true;

    return false;
  }

  if (casting === 'same_kind') {
    // Same kind: integers to integers, floats to floats, etc.
    // Does NOT allow cross-kind casting (int->float, float->complex)
    if (isIntegerDType(fromDtype) && isIntegerDType(to)) return true;
    if (isFloatDType(fromDtype) && isFloatDType(to)) return true;
    if (isComplexDType(fromDtype) && isComplexDType(to)) return true;
    if (fromDtype === 'bool' && isIntegerDType(to)) return true;
    if (fromDtype === 'bool' && to === 'bool') return true;

    // Float to complex is allowed (same numeric kind in broader sense)
    if (isFloatDType(fromDtype) && isComplexDType(to)) return true;

    return false;
  }

  return false;
}

/**
 * Return a scalar type common to input arrays
 *
 * This returns a floating-point type that can represent all input types.
 * Integers are promoted to floats that can hold their full range.
 */
export function common_type(...arrays: NDArrayCore[]): DType {
  if (arrays.length === 0) {
    return 'float64';
  }

  let hasComplex = false;
  let needsFloat64 = false;

  for (const arr of arrays) {
    const dtype = arr.dtype as DType;
    if (isComplexDType(dtype)) {
      hasComplex = true;
      if (dtype === 'complex128') {
        needsFloat64 = true;
      }
    } else if (isFloatDType(dtype)) {
      if (dtype === 'float64') {
        needsFloat64 = true;
      }
    } else if (isIntegerDType(dtype)) {
      // 64-bit integers require float64 (float32 can't represent all values)
      if (dtype === 'int64' || dtype === 'uint64' || dtype === 'int32' || dtype === 'uint32') {
        needsFloat64 = true;
      }
    }
  }

  if (hasComplex) {
    return needsFloat64 ? 'complex128' : 'complex64';
  }

  return needsFloat64 ? 'float64' : 'float32';
}

/**
 * Return the type that results from applying promotion rules
 *
 * NumPy-compatible type promotion:
 * - Integers promote to larger integers that can hold both ranges
 * - int32 + uint32 -> int64 (needs signed type that fits both)
 * - Integer + float -> float64 if integer is 32+ bits
 * - float32 + complex64 -> complex64
 * - float64 + complex64 -> complex128
 */
export function result_type(...arrays_and_dtypes: (NDArrayCore | DType)[]): DType {
  if (arrays_and_dtypes.length === 0) {
    return 'float64';
  }

  const dtypes: DType[] = arrays_and_dtypes.map((item) =>
    item instanceof NDArrayCore ? (item.dtype as DType) : item
  );

  // Check for complex
  const hasComplex = dtypes.some(isComplexDType);
  const hasFloat = dtypes.some(isFloatDType);
  const hasInteger = dtypes.some(isIntegerDType);

  // Complex promotion
  if (hasComplex) {
    const hasFloat64 = dtypes.some((d) => d === 'float64');
    const hasComplex128 = dtypes.some((d) => d === 'complex128');
    const has64BitInt = dtypes.some(
      (d) => d === 'int64' || d === 'uint64' || d === 'int32' || d === 'uint32'
    );
    if (hasComplex128 || hasFloat64 || has64BitInt) {
      return 'complex128';
    }
    return 'complex64';
  }

  // Float promotion
  if (hasFloat) {
    const hasFloat64 = dtypes.some((d) => d === 'float64');
    // Integer + float: promote based on integer size
    if (hasInteger) {
      const has32BitOrLarger = dtypes.some(
        (d) => d === 'int32' || d === 'uint32' || d === 'int64' || d === 'uint64'
      );
      if (has32BitOrLarger || hasFloat64) {
        return 'float64';
      }
    }
    return hasFloat64 ? 'float64' : 'float32';
  }

  // Pure integer promotion
  if (hasInteger) {
    const hasSigned = dtypes.some((d) => SIGNED_INTEGER_DTYPES.includes(d));
    const hasUnsigned = dtypes.some((d) => UNSIGNED_INTEGER_DTYPES.includes(d));

    // Find max bit width
    const bitWidths: Record<DType, number> = {
      bool: 8,
      int8: 8,
      uint8: 8,
      int16: 16,
      uint16: 16,
      int32: 32,
      uint32: 32,
      int64: 64,
      uint64: 64,
      float32: 0,
      float64: 0,
      complex64: 0,
      complex128: 0,
    };
    const maxBits = Math.max(...dtypes.map((d) => bitWidths[d] ?? 0));

    // Mixed signed/unsigned: promote to next larger signed type
    if (hasSigned && hasUnsigned) {
      if (maxBits >= 32) return 'int64';
      if (maxBits >= 16) return 'int32';
      return 'int16';
    }

    // All signed
    if (hasSigned) {
      if (maxBits >= 64) return 'int64';
      if (maxBits >= 32) return 'int32';
      if (maxBits >= 16) return 'int16';
      return 'int8';
    }

    // All unsigned
    if (maxBits >= 64) return 'uint64';
    if (maxBits >= 32) return 'uint32';
    if (maxBits >= 16) return 'uint16';
    return 'uint8';
  }

  // Bool only
  return 'bool';
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
      case 'inexact':
        // Inexact includes floats and complex
        return FLOAT_DTYPES.includes(dt1) || COMPLEX_DTYPES.includes(dt1);
      default:
        return dt1 === dtype2;
    }
  }

  return dt1 === dtype2;
}

/**
 * Return a description for the given dtype
 *
 * Returns the dtype name directly (NumPy compatible).
 */
export function typename(dtype: DType): string {
  return dtype;
}

/**
 * Return the character code of the minimum-size type needed
 *
 * Returns the minimum type from typeset that can safely represent all input types.
 * Uses safe casting rules to ensure no precision loss.
 */
export function mintypecode(
  typechars: string,
  typeset: string = 'GDFgdf',
  default_: string = 'd'
): string {
  // Map type characters to dtypes
  const charToDtype: Record<string, DType | null> = {
    b: 'int8',
    B: 'uint8',
    h: 'int16',
    H: 'uint16',
    i: 'int32',
    I: 'uint32',
    l: 'int64',
    L: 'uint64',
    f: 'float32',
    d: 'float64',
    F: 'complex64',
    D: 'complex128',
    g: 'float64', // long double maps to float64
    G: 'complex128', // complex long double maps to complex128
  };

  // Rank for ordering types (lower is smaller)
  // Use separate sub-ranks to prefer standard types over long-double aliases
  const typeCodeToRank: Record<string, number> = {
    b: 0,
    B: 1,
    h: 2,
    H: 3,
    i: 4,
    I: 5,
    l: 6,
    L: 7,
    f: 8.0,
    g: 9.1, // long double - slightly higher than d
    d: 9.0, // prefer d over g
    F: 10.0,
    G: 11.1, // complex long double - slightly higher than D
    D: 11.0, // prefer D over G
  };

  if (typechars.length === 0) {
    return default_;
  }

  // Get input dtypes
  const inputDtypes: DType[] = [];
  for (const char of typechars) {
    const dtype = charToDtype[char];
    if (dtype) {
      inputDtypes.push(dtype);
    }
  }

  if (inputDtypes.length === 0) {
    return default_;
  }

  // Find the minimum type in typeset that can safely represent ALL input types
  let bestChar = default_;
  let bestRank = Infinity;

  for (const char of typeset) {
    const targetDtype = charToDtype[char];
    if (!targetDtype) continue;

    const rank = typeCodeToRank[char] ?? Infinity;

    // Check if this type can safely cast from ALL inputs
    let canCastAll = true;
    for (const srcDtype of inputDtypes) {
      if (!can_cast(srcDtype, targetDtype, 'safe')) {
        canCastAll = false;
        break;
      }
    }

    if (canCastAll && rank < bestRank) {
      bestRank = rank;
      bestChar = char;
    }
  }

  return bestChar;
}
