/**
 * DType (Data Type) system for numpy-ts
 *
 * Supports NumPy numeric types:
 * - Floating point: float32, float64
 * - Complex: complex64, complex128
 * - Signed integers: int8, int16, int32, int64
 * - Unsigned integers: uint8, uint16, uint32, uint64
 * - Boolean: bool
 */

/**
 * All supported dtypes
 */
export type DType =
  | 'float64'
  | 'float32'
  | 'complex128'
  | 'complex64'
  | 'int64'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint64'
  | 'uint32'
  | 'uint16'
  | 'uint8'
  | 'bool';

/**
 * TypedArray types for each dtype
 */
export type TypedArray =
  | Float64Array
  | Float32Array
  | BigInt64Array
  | Int32Array
  | Int16Array
  | Int8Array
  | BigUint64Array
  | Uint32Array
  | Uint16Array
  | Uint8Array;

/**
 * Default dtype (matches NumPy)
 */
export const DEFAULT_DTYPE: DType = 'float64';

/**
 * Get the TypedArray constructor for a given dtype.
 * For complex types, returns the underlying float array constructor.
 * complex128 uses Float64Array, complex64 uses Float32Array.
 * Note: Complex arrays have 2x the physical elements (interleaved real, imag).
 */
export function getTypedArrayConstructor(dtype: DType): TypedArrayConstructor | null {
  switch (dtype) {
    case 'float64':
      return Float64Array;
    case 'float32':
      return Float32Array;
    case 'complex128':
      return Float64Array; // Interleaved: [re, im, re, im, ...]
    case 'complex64':
      return Float32Array; // Interleaved: [re, im, re, im, ...]
    case 'int64':
      return BigInt64Array;
    case 'int32':
      return Int32Array;
    case 'int16':
      return Int16Array;
    case 'int8':
      return Int8Array;
    case 'uint64':
      return BigUint64Array;
    case 'uint32':
      return Uint32Array;
    case 'uint16':
      return Uint16Array;
    case 'uint8':
      return Uint8Array;
    case 'bool':
      return Uint8Array; // bool is stored as uint8
    default:
      throw new Error(`Unknown dtype: ${dtype}`);
  }
}

type TypedArrayConstructor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | BigInt64ArrayConstructor
  | Int32ArrayConstructor
  | Int16ArrayConstructor
  | Int8ArrayConstructor
  | BigUint64ArrayConstructor
  | Uint32ArrayConstructor
  | Uint16ArrayConstructor
  | Uint8ArrayConstructor;

/**
 * Get the element size in bytes for a given dtype.
 * For complex types, returns the full element size (both real and imag parts).
 */
export function getDTypeSize(dtype: DType): number {
  switch (dtype) {
    case 'complex128':
      return 16; // Two float64 values
    case 'float64':
    case 'int64':
    case 'uint64':
    case 'complex64':
      return 8; // complex64 = two float32 values
    case 'float32':
    case 'int32':
    case 'uint32':
      return 4;
    case 'int16':
    case 'uint16':
      return 2;
    case 'int8':
    case 'uint8':
    case 'bool':
      return 1;
    default:
      throw new Error(`Unknown dtype: ${dtype}`);
  }
}

/**
 * Check if dtype is integer
 */
export function isIntegerDType(dtype: DType): boolean {
  return (
    dtype === 'int64' ||
    dtype === 'int32' ||
    dtype === 'int16' ||
    dtype === 'int8' ||
    dtype === 'uint64' ||
    dtype === 'uint32' ||
    dtype === 'uint16' ||
    dtype === 'uint8'
  );
}

/**
 * Check if dtype is floating point
 */
export function isFloatDType(dtype: DType): boolean {
  return dtype === 'float64' || dtype === 'float32';
}

/**
 * Check if dtype uses BigInt
 */
export function isBigIntDType(dtype: DType): boolean {
  return dtype === 'int64' || dtype === 'uint64';
}

/**
 * Check if dtype is complex
 */
export function isComplexDType(dtype: DType): boolean {
  return dtype === 'complex64' || dtype === 'complex128';
}

/**
 * Get the underlying float dtype for a complex dtype.
 * complex128 -> float64, complex64 -> float32
 */
export function getComplexComponentDType(dtype: DType): 'float64' | 'float32' {
  if (dtype === 'complex128') return 'float64';
  if (dtype === 'complex64') return 'float32';
  throw new Error(`${dtype} is not a complex dtype`);
}

/**
 * Get the complex dtype for a given float component dtype.
 * float64 -> complex128, float32 -> complex64
 */
export function getComplexDType(componentDtype: 'float64' | 'float32'): 'complex128' | 'complex64' {
  if (componentDtype === 'float64') return 'complex128';
  if (componentDtype === 'float32') return 'complex64';
  throw new Error(`${componentDtype} is not a valid complex component dtype`);
}

/**
 * Check if a value looks like a complex number input.
 * Accepts: Complex instance, {re, im} object
 */
export function isComplexLike(value: unknown): boolean {
  if (typeof value === 'object' && value !== null) {
    // Check for Complex class or {re, im} object
    if ('re' in value && 'im' in value) {
      return true;
    }
  }
  return false;
}

/**
 * Infer dtype from JavaScript value
 */
export function inferDType(value: unknown): DType {
  if (typeof value === 'bigint') {
    return 'int64';
  } else if (typeof value === 'number') {
    // Check if integer
    if (Number.isInteger(value)) {
      // Choose appropriate integer type based on range
      if (value >= 0 && value <= 255) return 'uint8';
      if (value >= -128 && value <= 127) return 'int8';
      if (value >= 0 && value <= 65535) return 'uint16';
      if (value >= -32768 && value <= 32767) return 'int16';
      if (value >= 0 && value <= 4294967295) return 'uint32';
      if (value >= -2147483648 && value <= 2147483647) return 'int32';
      return 'float64'; // Fallback for large integers
    }
    return 'float64';
  } else if (typeof value === 'boolean') {
    return 'bool';
  } else if (isComplexLike(value)) {
    return 'complex128'; // Default complex dtype
  }
  return DEFAULT_DTYPE;
}

/**
 * Promote two dtypes to a common dtype
 * Follows NumPy's type promotion rules
 */
export function promoteDTypes(dtype1: DType, dtype2: DType): DType {
  // Same dtype
  if (dtype1 === dtype2) return dtype1;

  // Boolean - promote to the other type
  if (dtype1 === 'bool') return dtype2;
  if (dtype2 === 'bool') return dtype1;

  // Complex types - complex always wins over real types
  // NumPy promotion rules:
  // - complex64 + float32 → complex64
  // - complex64 + float64 → complex128
  // - complex128 + any_real → complex128
  // - complex64 + complex128 → complex128
  if (isComplexDType(dtype1) || isComplexDType(dtype2)) {
    // Both complex
    if (isComplexDType(dtype1) && isComplexDType(dtype2)) {
      // complex128 wins over complex64
      if (dtype1 === 'complex128' || dtype2 === 'complex128') {
        return 'complex128';
      }
      return 'complex64';
    }

    // One complex, one real
    const complexDtype = isComplexDType(dtype1) ? dtype1 : dtype2;
    const realDtype = isComplexDType(dtype1) ? dtype2 : dtype1;

    // complex128 + anything → complex128
    if (complexDtype === 'complex128') {
      return 'complex128';
    }

    // complex64 + float64 → complex128
    // complex64 + int64/uint64 → complex128
    // complex64 + int32/uint32 → complex128 (since float32 can't hold all int32 values)
    if (
      realDtype === 'float64' ||
      realDtype === 'int64' ||
      realDtype === 'uint64' ||
      realDtype === 'int32' ||
      realDtype === 'uint32'
    ) {
      return 'complex128';
    }

    // complex64 + float32 or smaller ints → complex64
    return 'complex64';
  }

  // Float types - integer + float always promotes to float
  if (isFloatDType(dtype1) || isFloatDType(dtype2)) {
    // NumPy behavior: Promote to the float type
    // float64 always wins
    if (dtype1 === 'float64' || dtype2 === 'float64') return 'float64';

    // float32 with small integers (8, 16 bit) → float32
    // float32 with large integers (32, 64 bit) → float64 (precision safety)
    // This is because float32 has 24-bit mantissa, can't hold all int32 values
    if (dtype1 === 'float32') {
      const intDtype = dtype2;
      if (
        intDtype === 'int32' ||
        intDtype === 'int64' ||
        intDtype === 'uint32' ||
        intDtype === 'uint64'
      ) {
        return 'float64';
      }
      return 'float32';
    }
    if (dtype2 === 'float32') {
      const intDtype = dtype1;
      if (
        intDtype === 'int32' ||
        intDtype === 'int64' ||
        intDtype === 'uint32' ||
        intDtype === 'uint64'
      ) {
        return 'float64';
      }
      return 'float32';
    }

    // Both are float32
    return 'float32';
  }

  // Integer types - complex promotion rules
  const isSigned1 = dtype1.startsWith('int');
  const isSigned2 = dtype2.startsWith('int');
  const isUnsigned1 = dtype1.startsWith('uint');
  const isUnsigned2 = dtype2.startsWith('uint');

  // Get bit sizes
  const getSize = (dtype: DType): number => {
    if (dtype.includes('64')) return 64;
    if (dtype.includes('32')) return 32;
    if (dtype.includes('16')) return 16;
    if (dtype.includes('8')) return 8;
    return 0;
  };

  const size1 = getSize(dtype1);
  const size2 = getSize(dtype2);

  // Special case: int64 + uint64 → float64 (no larger int type available)
  if ((dtype1 === 'int64' && dtype2 === 'uint64') || (dtype1 === 'uint64' && dtype2 === 'int64')) {
    return 'float64';
  }

  // Mixing signed and unsigned of the same size: promote to larger signed type
  if (isSigned1 && isUnsigned2 && size1 === size2) {
    if (size1 === 8) return 'int16';
    if (size1 === 16) return 'int32';
    if (size1 === 32) return 'int64';
  }
  if (isUnsigned1 && isSigned2 && size1 === size2) {
    if (size2 === 8) return 'int16';
    if (size2 === 16) return 'int32';
    if (size2 === 32) return 'int64';
  }

  // Same signedness: promote to larger size
  if ((isSigned1 && isSigned2) || (isUnsigned1 && isUnsigned2)) {
    const maxSize = Math.max(size1, size2);
    if (isSigned1) {
      if (maxSize === 64) return 'int64';
      if (maxSize === 32) return 'int32';
      if (maxSize === 16) return 'int16';
      return 'int8';
    } else {
      if (maxSize === 64) return 'uint64';
      if (maxSize === 32) return 'uint32';
      if (maxSize === 16) return 'uint16';
      return 'uint8';
    }
  }

  // Different signedness, different sizes: promote to type that can hold both
  // NumPy behavior: If signed type is larger, use it; otherwise promote conservatively

  // If signed int is larger than unsigned int, signed int can hold unsigned
  if (isSigned1 && isUnsigned2) {
    if (size1 > size2) {
      // e.g., int16 + uint8 → int16, int32 + uint16 → int32
      return dtype1;
    }
    // Otherwise need larger signed type
    // e.g., int8 + uint16 → int32
    if (size2 === 8) return 'int16';
    if (size2 === 16) return 'int32';
    if (size2 === 32) return 'int64';
    return 'float64'; // uint64 with smaller signed → float64
  }

  if (isUnsigned1 && isSigned2) {
    if (size2 > size1) {
      // e.g., uint8 + int16 → int16, uint16 + int32 → int32
      return dtype2;
    }
    // Otherwise need larger signed type
    // e.g., uint16 + int8 → int32
    if (size1 === 8) return 'int16';
    if (size1 === 16) return 'int32';
    if (size1 === 32) return 'int64';
    return 'float64'; // uint64 with smaller signed → float64
  }

  // Fallback (shouldn't reach here if logic above is complete)
  return 'float64';
}

/**
 * Validate dtype string
 */
export function isValidDType(dtype: string): dtype is DType {
  const validDTypes: DType[] = [
    'float64',
    'float32',
    'complex128',
    'complex64',
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
  return validDTypes.includes(dtype as DType);
}

/**
 * Convert value to the appropriate type for the given dtype
 */
export function castValue(value: number | bigint | boolean, dtype: DType): number | bigint {
  if (isBigIntDType(dtype)) {
    return BigInt(value as number | bigint);
  }
  if (dtype === 'bool') {
    return value ? 1 : 0;
  }
  return Number(value);
}

/**
 * Map our dtype to @stdlib's supported dtype
 * @stdlib doesn't support int64, uint64 natively, but does support bool
 */
export function toStdlibDType(dtype: DType): string {
  // Map int64/uint64 to generic (we manage the BigInt arrays ourselves)
  if (dtype === 'int64' || dtype === 'uint64') {
    return 'generic';
  }
  // Map complex to generic (we manage complex arrays ourselves)
  if (isComplexDType(dtype)) {
    return 'generic';
  }
  // All other dtypes (including bool) are supported by stdlib
  return dtype;
}
