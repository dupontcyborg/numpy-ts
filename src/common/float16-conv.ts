/**
 * IEEE 754 half-precision (float16) conversion utilities.
 *
 * Used for NPY serialization/deserialization when native Float16Array
 * is not available. When Float16Array IS available, these are only needed
 * for NPY byte-level operations.
 *
 * @internal
 */

// Reusable DataView for float32 bit manipulation
const f32Buf = new ArrayBuffer(4);
const f32View = new DataView(f32Buf);

/**
 * Convert a float16 (stored as uint16 bits) to a float64 number.
 * Handles denormals, infinities, NaN, and signed zero.
 */
export function float16BitsToNumber(bits: number): number {
  const sign = (bits >>> 15) & 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;

  let value: number;

  if (exponent === 0) {
    // Denormal or zero
    value = mantissa === 0 ? 0 : mantissa * 2 ** -24; // 2^(-14) * (mantissa / 1024)
  } else if (exponent === 31) {
    // Infinity or NaN
    value = mantissa === 0 ? Infinity : NaN;
  } else {
    // Normalized
    value = (mantissa / 1024 + 1) * 2 ** (exponent - 15);
  }

  return sign ? -value : value;
}

/**
 * Convert a float64 number to float16 bits (uint16).
 * Rounds to nearest even. Handles overflow to infinity and underflow to zero/denormal.
 */
export function numberToFloat16Bits(value: number): number {
  // Use Float32 bit representation as intermediate
  f32View.setFloat32(0, value, false);
  const f32Bits = f32View.getUint32(0, false);

  const sign = (f32Bits >>> 31) & 1;
  const exponent = (f32Bits >>> 23) & 0xff;
  const mantissa = f32Bits & 0x7fffff;

  let f16Exp: number;
  let f16Man: number;

  if (exponent === 0xff) {
    // Infinity or NaN
    f16Exp = 0x1f;
    f16Man = mantissa === 0 ? 0 : 0x200; // Preserve NaN (any non-zero mantissa)
  } else if (exponent > 142) {
    // Overflow → infinity (float32 exp 142 = float16 exp 30, the max normal)
    f16Exp = 0x1f;
    f16Man = 0;
  } else if (exponent < 103) {
    // Underflow → zero (too small for even denormal float16)
    f16Exp = 0;
    f16Man = 0;
  } else if (exponent < 113) {
    // Denormal float16
    f16Exp = 0;
    // Add implicit 1 bit and shift right
    const shift = 125 - exponent; // 113 - exponent + 12 (13 bits → 10 bits + hidden bit)
    f16Man = (0x800000 | mantissa) >>> shift;
    // Round to nearest even
    const remainder = (0x800000 | mantissa) & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (f16Man & 1) !== 0)) {
      f16Man++;
    }
  } else {
    // Normal float16
    f16Exp = exponent - 112; // Bias adjustment: float32 bias 127 → float16 bias 15
    f16Man = mantissa >>> 13;
    // Round to nearest even
    const remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (f16Man & 1) !== 0)) {
      f16Man++;
      if (f16Man > 0x3ff) {
        f16Man = 0;
        f16Exp++;
        if (f16Exp > 0x1e) {
          // Overflow to infinity
          f16Exp = 0x1f;
          f16Man = 0;
        }
      }
    }
  }

  return (sign << 15) | (f16Exp << 10) | f16Man;
}

/**
 * Create a typed array from float16 raw bytes (for NPY parsing).
 *
 * If native Float16Array is available, constructs one directly.
 * Otherwise, converts each float16 value to float32 and returns a Float32Array.
 */
export function float16BytesToTypedArray(
  buffer: ArrayBuffer,
  byteOffset: number,
  numElements: number,
  needsByteSwap: boolean
): Float32Array {
  const view = new DataView(buffer, byteOffset);
  const result = new Float32Array(numElements);

  for (let i = 0; i < numElements; i++) {
    let bits: number;
    if (needsByteSwap) {
      // Read little-endian since the caller handles byte swapping before us.
      bits = view.getUint16(i * 2, true);
    } else {
      bits = view.getUint16(i * 2, true); // Little-endian (NPY default)
    }
    result[i] = float16BitsToNumber(bits);
  }

  return result;
}
