/**
 * Type checking and dtype utility functions
 *
 * Tree-shakeable standalone functions for dtype operations.
 */

import {
  isComplexDType,
  isFloatDType,
  isIntegerDType,
  isValidDType,
  promoteDTypeSequence,
  promoteDTypes,
} from '../common/dtype';
import { type DType, NDArrayCore } from '../common/ndarray-core';

// Dtype categories for type checking
const SIGNED_INTEGER_DTYPES: DType[] = ['int8', 'int16', 'int32', 'int64'];
const UNSIGNED_INTEGER_DTYPES: DType[] = ['uint8', 'uint16', 'uint32', 'uint64'];
const INTEGER_DTYPES: DType[] = [...SIGNED_INTEGER_DTYPES, ...UNSIGNED_INTEGER_DTYPES];
const FLOAT_DTYPES: DType[] = ['float16', 'float32', 'float64'];
const COMPLEX_DTYPES: DType[] = ['complex64', 'complex128'];
const NUMERIC_DTYPES: DType[] = [...INTEGER_DTYPES, ...FLOAT_DTYPES, ...COMPLEX_DTYPES];

// Kind order used by same_kind casting: a cast is allowed toward an equal or later kind.
const KIND_ORDER: Record<DType, number> = {
  bool: 0,
  uint8: 1,
  uint16: 1,
  uint32: 1,
  uint64: 1,
  int8: 2,
  int16: 2,
  int32: 2,
  int64: 2,
  float16: 3,
  float32: 3,
  float64: 3,
  complex64: 4,
  complex128: 4,
};

function assertDType(d: unknown): asserts d is DType {
  if (typeof d !== 'string' || !isValidDType(d)) {
    throw new TypeError(`data type '${String(d)}' not understood`);
  }
}

/**
 * Check if a cast between dtypes is possible according to the casting rule. Safe casting
 * means the target holds every source value (promotion of the pair is the target);
 * same_kind also allows casts toward an equal or later kind in bool, unsigned, signed,
 * float, complex order.
 */
export function can_cast(
  from_: DType | NDArrayCore,
  to: DType,
  casting: 'no' | 'equiv' | 'safe' | 'same_kind' | 'unsafe' = 'safe',
): boolean {
  const fromDtype: DType = from_ instanceof NDArrayCore ? (from_.dtype as DType) : from_;
  assertDType(fromDtype);
  assertDType(to);

  switch (casting) {
    case 'no':
    case 'equiv':
      return fromDtype === to;
    case 'unsafe':
      return true;
    case 'safe':
      return promoteDTypes(fromDtype, to) === to;
    case 'same_kind':
      return promoteDTypes(fromDtype, to) === to || KIND_ORDER[fromDtype] <= KIND_ORDER[to];
    default:
      throw new Error("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'");
  }
}

/**
 * Returns a scalar type common to the input arrays: a floating-point type that can
 * represent all of them, with integers promoted to floats wide enough for their range.
 */
export function common_type(...arrays: NDArrayCore[]): DType {
  if (arrays.length === 0) {
    return 'float64';
  }

  let hasComplex = false;
  let needsFloat64 = false;
  let needsFloat32 = false;

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
      } else if (dtype === 'float32') {
        needsFloat32 = true;
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

  if (needsFloat64) return 'float64';
  // common_type returns a float that can represent all inputs.
  // For integer-only inputs, float32 is the minimum (matching NumPy).
  // float16 is only returned when all inputs are already float16.
  if (needsFloat32) return 'float32';
  const allFloat16 = arrays.every((a) => a.dtype === 'float16');
  return allFloat16 ? 'float16' : 'float32';
}

/**
 * Returns the NumPy type promotion of all inputs. Several inputs are not promoted by a
 * plain left fold: NumPy promotes the combined float or complex dtype with each integer
 * separately, so result_type of int8, uint16 and float32 is float32.
 *
 * @param arrays_and_dtypes - Arrays or dtype names; at least one is required
 * @returns Promoted dtype
 */
export function result_type(...arrays_and_dtypes: (NDArrayCore | DType)[]): DType {
  if (arrays_and_dtypes.length === 0) {
    throw new Error('at least one array or dtype is required');
  }

  const dtypes: DType[] = arrays_and_dtypes.map((item) => {
    const d = item instanceof NDArrayCore ? (item.dtype as DType) : item;
    assertDType(d);
    return d;
  });

  return promoteDTypeSequence(dtypes);
}

/**
 * Return the smallest dtype that can hold a scalar value. Integers use the smallest
 * unsigned type for non-negative values and the smallest signed type for negative ones.
 * Non-integer numbers pick float16, float32 or float64 by magnitude alone, as NumPy
 * does. A JS number that is integer-valued is treated as an integer, since JS cannot
 * distinguish 1.0 from 1. Integers outside the 64-bit range have no dtype here and throw.
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
      if (val <= 18446744073709551615n) return 'uint64';
    } else {
      if (val >= -128n) return 'int8';
      if (val >= -32768n) return 'int16';
      if (val >= -2147483648n) return 'int32';
      if (val >= -9223372036854775808n) return 'int64';
    }
    throw new RangeError(`min_scalar_type: ${val} does not fit in any supported integer dtype`);
  }

  if (Number.isInteger(val) && !Object.is(val, -0)) {
    if (val >= 0) {
      if (val <= 255) return 'uint8';
      if (val <= 65535) return 'uint16';
      if (val <= 4294967295) return 'uint32';
      if (val < 2 ** 64) return 'uint64';
    } else {
      if (val >= -128) return 'int8';
      if (val >= -32768) return 'int16';
      if (val >= -2147483648) return 'int32';
      if (val >= -(2 ** 63)) return 'int64';
    }
  }

  // NumPy gives NaN and both infinities float16.
  if (!Number.isFinite(val)) return 'float16';
  const magnitude = Math.abs(val);
  if (magnitude < 65000) return 'float16';
  return magnitude < 3.4e38 ? 'float32' : 'float64';
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

const TYPE_NAMES = new Map<string, string>([
  ['S1', 'character'],
  ['?', 'bool'],
  ['b', 'signed char'],
  ['B', 'unsigned char'],
  ['h', 'short'],
  ['H', 'unsigned short'],
  ['i', 'integer'],
  ['I', 'unsigned integer'],
  ['l', 'long integer'],
  ['L', 'unsigned long integer'],
  ['q', 'long long integer'],
  ['Q', 'unsigned long long integer'],
  ['f', 'single precision'],
  ['d', 'double precision'],
  ['g', 'long precision'],
  ['F', 'complex single precision'],
  ['D', 'complex double precision'],
  ['G', 'complex long double precision'],
  ['S', 'string'],
  ['U', 'unicode'],
  ['V', 'void'],
  ['O', 'object'],
]);

/**
 * Returns NumPy's description of a type character code, for example 'd' gives
 * 'double precision' and 'S1' gives 'character'. A code NumPy has no description for
 * throws.
 *
 * @param char - Type character code
 * @returns Description of the type code
 */
export function typename(char: string): string {
  const name = TYPE_NAMES.get(char);
  if (name === undefined) {
    throw new Error(`typename: unknown type code '${String(char)}'`);
  }
  return name;
}

// Type codes ordered largest first; mintypecode picks the earliest one present.
const TYPECODES_BY_ELSIZE = 'GDFgdfQqLlIiHhBb?';

/**
 * Returns the character code for the smallest type in typeset that covers the given
 * type codes, following NumPy: each element of typechars is a whole code (a plain string
 * is read character by character), only codes found in typeset count, 'F' together
 * with 'd' gives 'D', and the default is returned when nothing matches. Membership in a
 * string typeset is a substring test, as in NumPy.
 *
 * @param typechars - String of type codes or array of type-code strings
 * @param typeset - String or array of the codes the result is chosen from
 * @param default_ - Result when no input code is in typeset
 * @returns The selected type code
 */
export function mintypecode(
  typechars: string | readonly string[],
  typeset: string | readonly string[] = 'GDFgdf',
  default_: string = 'd',
): string {
  const intersection = new Set<string>();
  for (const item of typechars) {
    // An empty string has no code of its own; NumPy reads it as a unicode array, 'U'.
    const t = item === '' ? 'U' : item;
    if (typeset.includes(t)) intersection.add(t);
  }
  if (intersection.size === 0) return default_;
  if (intersection.has('F') && intersection.has('d')) return 'D';

  let best: string | undefined;
  let bestIdx = Infinity;
  for (const t of intersection) {
    const idx = TYPECODES_BY_ELSIZE.indexOf(t);
    if (idx === -1) throw new Error(`mintypecode: '${t}' is not a known type code`);
    if (idx < bestIdx) {
      bestIdx = idx;
      best = t;
    }
  }
  return best as string;
}
