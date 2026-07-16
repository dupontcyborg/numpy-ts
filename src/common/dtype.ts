/**
 * DType (Data Type) system for numpy-ts
 *
 * Supports NumPy numeric types:
 * - Floating point: float16, float32, float64
 * - Complex: complex64, complex128
 * - Signed integers: int8, int16, int32, int64
 * - Unsigned integers: uint8, uint16, uint32, uint64
 * - Boolean: bool
 */

import type { Complex } from './complex';

/**
 * All supported dtypes, in canonical declaration order.
 *
 * This array is the single source of truth for the dtype list: the {@link DType}
 * union is derived from it, and codegen (`scripts/generate-dtype-promotion.ts`)
 * plus the oracle sweeps import it so a new dtype only has to be added here.
 */
export const DTYPES = [
  'float64',
  'float32',
  'float16',
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
] as const;

/**
 * All supported dtypes
 */
export type DType = (typeof DTYPES)[number];

/**
 * Dtypes backed by BigInt typed arrays (element access yields `bigint`).
 */
export type BigIntDType = 'int64' | 'uint64';

/**
 * Complex dtypes (element access yields a `Complex`).
 */
export type ComplexDType = 'complex64' | 'complex128';

/**
 * The JavaScript scalar type produced by element access (`.get()`, `.item()`,
 * `.iget()`) for a given dtype:
 *   - int64/uint64    → bigint
 *   - complex64/128   → Complex
 *   - everything else → number
 *
 * When `D` is the full `DType` union (the default), this distributes to
 * `number | bigint | Complex`, matching the historical untyped return.
 */
export type Scalar<D extends DType = DType> = D extends BigIntDType
  ? bigint
  : D extends ComplexDType
    ? Complex
    : number;

/** Numeric kind of a dtype. */
type DTypeKind = 'float' | 'complex' | 'int' | 'uint' | 'bool';

/** Structural metadata (kind + bit width) for each dtype. */
interface DTypeMeta {
  kind: DTypeKind;
  bits: number;
}

/**
 * Per-dtype structural metadata. Keyed by the full `DType` union, so adding a
 * dtype to {@link DTYPES} without describing it here is a compile-time error —
 * this is what lets {@link promoteDTypes} classify integers by data instead of
 * brittle string matching (`.startsWith('int')` / `.includes('64')`).
 */
const DTYPE_META: Record<DType, DTypeMeta> = {
  float64: { kind: 'float', bits: 64 },
  float32: { kind: 'float', bits: 32 },
  float16: { kind: 'float', bits: 16 },
  complex128: { kind: 'complex', bits: 128 },
  complex64: { kind: 'complex', bits: 64 },
  int64: { kind: 'int', bits: 64 },
  int32: { kind: 'int', bits: 32 },
  int16: { kind: 'int', bits: 16 },
  int8: { kind: 'int', bits: 8 },
  uint64: { kind: 'uint', bits: 64 },
  uint32: { kind: 'uint', bits: 32 },
  uint16: { kind: 'uint', bits: 16 },
  uint8: { kind: 'uint', bits: 8 },
  bool: { kind: 'bool', bits: 8 },
};

/**
 * TypedArray types for each dtype
 */
export type TypedArray =
  | Float64Array
  | Float32Array
  | Float16Array
  | BigInt64Array
  | Int32Array
  | Int16Array
  | Int8Array
  | BigUint64Array
  | Uint32Array
  | Uint16Array
  | Uint8Array;

/**
 * Whether the runtime supports native Float16Array (TC39 proposal, available in modern engines).
 * When false, float16 dtype uses Float32Array as a fallback backing store.
 */
export const hasFloat16: boolean = typeof globalThis.Float16Array !== 'undefined';

/**
 * Returns the effective dtype for WASM kernel dispatch.
 * When Float16Array is unavailable, float16 storage is backed by Float32Array,
 * so WASM kernels should treat it as float32.
 */
export function effectiveDType(dtype: DType): DType {
  return dtype === 'float16' && !hasFloat16 ? 'float32' : dtype;
}

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
    case 'float16':
      // Use native Float16Array when available, otherwise fall back to Float32Array
      return hasFloat16 ? Float16Array : Float32Array;
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
  | Float16ArrayConstructor
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
    case 'float16':
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
  return dtype === 'float64' || dtype === 'float32' || dtype === 'float16';
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
 * Throw a TypeError if the dtype is complex.
 * Use this at the start of functions that don't support complex numbers.
 *
 * @param dtype - The dtype to check
 * @param functionName - The name of the function (for error message)
 * @param reason - Optional reason why complex is not supported
 * @throws TypeError if dtype is complex
 *
 * @example
 * ```typescript
 * export function floor(storage: ArrayStorage): ArrayStorage {
 *   throwIfComplex(storage.dtype, 'floor', 'rounding is not defined for complex numbers');
 *   // ... rest of implementation
 * }
 * ```
 */
export function throwIfComplex(dtype: DType, functionName: string, reason?: string): void {
  if (isComplexDType(dtype)) {
    const reasonStr = reason ? ` ${reason}` : '';
    throw new TypeError(
      `ufunc '${functionName}' not supported for complex dtype '${dtype}'.${reasonStr}`,
    );
  }
}

/**
 * Throw a TypeError if the dtype is bool and the operation doesn't support it.
 * Matches NumPy 2.x behavior where boolean arithmetic (negative, subtract, etc.) is rejected.
 */
export function throwIfBool(dtype: DType, functionName: string, reason?: string): void {
  if (dtype === 'bool') {
    const reasonStr = reason ? ` ${reason}` : '';
    throw new TypeError(`ufunc '${functionName}' not supported for boolean dtype.${reasonStr}`);
  }
}

/**
 * Throw an error if the dtype is complex and the function doesn't yet support it.
 * Use this for functions that SHOULD support complex but haven't been implemented yet.
 *
 * @param dtype - The dtype to check
 * @param functionName - The name of the function (for error message)
 * @throws Error if dtype is complex
 *
 * @example
 * ```typescript
 * export function sin(storage: ArrayStorage): ArrayStorage {
 *   throwIfComplexNotImplemented(storage.dtype, 'sin');
 *   // ... existing real-only implementation
 * }
 * ```
 */
export function throwIfComplexNotImplemented(dtype: DType, functionName: string): void {
  if (isComplexDType(dtype)) {
    throw new Error(
      `'${functionName}' does not yet support complex dtype '${dtype}'. ` +
        `Complex support is planned but not yet implemented.`,
    );
  }
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

    // float32 wins over float16
    if (dtype1 === 'float32' || dtype2 === 'float32') {
      // float32 with large integers (32, 64 bit) → float64 (precision safety)
      // This is because float32 has 24-bit mantissa, can't hold all int32 values
      const otherDtype = dtype1 === 'float32' ? dtype2 : dtype1;
      if (
        otherDtype === 'int32' ||
        otherDtype === 'int64' ||
        otherDtype === 'uint32' ||
        otherDtype === 'uint64'
      ) {
        return 'float64';
      }
      return 'float32';
    }

    // float16 + float16 = float16
    // float16 with small integers (8 bit) → float16 (float16 has 11-bit mantissa)
    // float16 with 16-bit+ integers → float32 (precision safety)
    if (dtype1 === 'float16' || dtype2 === 'float16') {
      const otherDtype = dtype1 === 'float16' ? dtype2 : dtype1;
      if (otherDtype === 'float16' || otherDtype === 'int8' || otherDtype === 'uint8') {
        return 'float16';
      }
      // int16, uint16, int32, uint32, int64, uint64 → need at least float32
      if (otherDtype === 'int16' || otherDtype === 'uint16') {
        return 'float32';
      }
      // int32+ → float64
      return 'float64';
    }

    // Both are float32 (unreachable now but kept for safety)
    return 'float32';
  }

  // Integer types - complex promotion rules. Classify by structural metadata
  // rather than string matching so a new dtype can't silently mis-size.
  const meta1 = DTYPE_META[dtype1];
  const meta2 = DTYPE_META[dtype2];
  const isSigned1 = meta1.kind === 'int';
  const isSigned2 = meta2.kind === 'int';
  const isUnsigned1 = meta1.kind === 'uint';
  const isUnsigned2 = meta2.kind === 'uint';

  const size1 = meta1.bits;
  const size2 = meta2.bits;

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

  // Unreachable: every integer pair is handled above. Throw loudly if a new
  // dtype slips through unhandled rather than silently coercing to float64.
  throw new Error(`promoteDTypes: unhandled dtype pair '${dtype1}' + '${dtype2}'`);
}

// ─── Unary / reduction result-dtype rules (NumPy 2.x) ───────────────────────

/**
 * Result dtype for unary math functions (sin, cos, sqrt, exp, log, etc.).
 * Maps input dtype to the "minimum float" that can represent it.
 *
 * bool/int8/uint8  → float16 (if available, else float32)
 * int16/uint16     → float32
 * int32+/uint32+   → float64
 * float/complex    → same dtype (preserved)
 */
export function mathResultDtype(inputDtype: DType): DType {
  const canonical = mathResultDtypeCanonical(inputDtype);
  // Runtime fallback: float16 storage is backed by Float32Array when the engine
  // lacks native Float16Array, so the result dtype degrades to float32 there.
  return canonical === 'float16' && !hasFloat16 ? 'float32' : canonical;
}

/**
 * Canonical form of {@link mathResultDtype}, assuming native `Float16Array` is
 * available (bool/int8/uint8 → float16). This is what the compile-time type
 * tables are generated from, so the types stay canonical regardless of the
 * engine the generator runs on. The runtime {@link mathResultDtype} applies the
 * float32 fallback on top of this.
 */
export function mathResultDtypeCanonical(inputDtype: DType): DType {
  switch (inputDtype) {
    case 'bool':
    case 'int8':
    case 'uint8':
      return 'float16';
    case 'int16':
    case 'uint16':
      return 'float32';
    case 'int32':
    case 'uint32':
    case 'int64':
    case 'uint64':
      return 'float64';
    default:
      return inputDtype; // float16/32/64, complex64/128 preserved
  }
}

/**
 * Result dtype for bool inputs in arithmetic ops that promote bool → int8.
 * Used by: power, mod, floor_divide, remainder, square.
 */
export function boolArithmeticDtype(inputDtype: DType): DType {
  return inputDtype === 'bool' ? 'int8' : inputDtype;
}

/**
 * Result dtype for reduction accumulation (sum, prod, cumsum, cumprod).
 * NumPy accumulates small integers in wider types to prevent overflow.
 *
 * bool/int8/int16/int32  → int64
 * uint8/uint16/uint32    → uint64
 * int64/uint64           → same
 * float/complex          → same
 */
export function reductionAccumDtype(inputDtype: DType): DType {
  switch (inputDtype) {
    case 'bool':
    case 'int8':
    case 'int16':
    case 'int32':
      return 'int64';
    case 'uint8':
    case 'uint16':
    case 'uint32':
      return 'uint64';
    default:
      return inputDtype;
  }
}

/**
 * Result dtype for FFT functions.
 * float32 → complex64, everything else → complex128.
 */
export function fftResultDtype(inputDtype: DType): DType {
  if (inputDtype === 'float16' || inputDtype === 'float32' || inputDtype === 'complex64')
    return 'complex64';
  return 'complex128';
}

/**
 * Result dtype for real-output FFT functions (hfft).
 * float16/float32/complex64 → float32, everything else → float64.
 */
export function fftRealResultDtype(inputDtype: DType): DType {
  if (inputDtype === 'float16' || inputDtype === 'float32' || inputDtype === 'complex64')
    return 'float32';
  return 'float64';
}

// ─── Result-dtype rules: SSOT for the compile-time type tables ──────────────

/**
 * Result dtype for true division (`/`, divide, mean, median).
 * bool + all integers → float64; float/complex preserved.
 */
export function trueDivideResultDtype(inputDtype: DType): DType {
  if (isComplexDType(inputDtype) || isFloatDType(inputDtype)) return inputDtype;
  return 'float64';
}

/**
 * Result dtype for std/var (spread statistics).
 * Like {@link trueDivideResultDtype} but complex collapses to its real component
 * (complex128 → float64, complex64 → float32).
 */
export function stdVarResultDtype(inputDtype: DType): DType {
  if (inputDtype === 'complex128') return 'float64';
  if (inputDtype === 'complex64') return 'float32';
  if (isFloatDType(inputDtype)) return inputDtype;
  return 'float64';
}

/**
 * Result dtype for `absolute`. Complex magnitude is real (complex128 → float64,
 * complex64 → float32); every other dtype is preserved.
 */
export function absResultDtype(inputDtype: DType): DType {
  if (inputDtype === 'complex128') return 'float64';
  if (inputDtype === 'complex64') return 'float32';
  return inputDtype;
}

/**
 * Result dtype for `angle` (always real float), matching NumPy:
 *   - complex → its real component (complex64 → float32, complex128 → float64)
 *   - real floats → preserved
 *   - integers → the unary-math float promotion (int8/uint8 → float16, …)
 *   - bool → float64 (NumPy quirk: unlike other unary math, bool does not map to float16)
 */
export function angleResultDtype(inputDtype: DType): DType {
  if (isComplexDType(inputDtype)) return getComplexComponentDType(inputDtype);
  if (inputDtype === 'bool') return 'float64';
  return mathResultDtype(inputDtype);
}

/**
 * Canonical form of {@link angleResultDtype} (assumes native `Float16Array`).
 * Used by the type-table codegen so the generated `Angle<D>` stays canonical.
 */
export function angleResultDtypeCanonical(inputDtype: DType): DType {
  if (isComplexDType(inputDtype)) return getComplexComponentDType(inputDtype);
  if (inputDtype === 'bool') return 'float64';
  return mathResultDtypeCanonical(inputDtype);
}

/**
 * Result dtype for `round`/`around`. Preserves the input dtype for everything
 * except bool, which NumPy upcasts to float16 (rounding a boolean yields a
 * float). Every other dtype — ints, floats, complex — is returned unchanged.
 */
export function roundResultDtype(inputDtype: DType): DType {
  return inputDtype === 'bool' ? mathResultDtype('bool') : inputDtype;
}

/**
 * Canonical form of {@link roundResultDtype} (bool → float16). Used by the
 * type-table codegen so the generated `RoundResult<D>` stays float16-canonical.
 */
export function roundResultDtypeCanonical(inputDtype: DType): DType {
  return inputDtype === 'bool' ? 'float16' : inputDtype;
}

/**
 * Validate dtype string
 */
export function isValidDType(dtype: string): dtype is DType {
  return (DTYPES as readonly string[]).includes(dtype);
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
  // Map float16 to generic (stdlib doesn't support float16)
  if (dtype === 'float16') {
    return 'generic';
  }
  // Map complex to generic (we manage complex arrays ourselves)
  if (isComplexDType(dtype)) {
    return 'generic';
  }
  // All other dtypes (including bool) are supported by stdlib
  return dtype;
}
