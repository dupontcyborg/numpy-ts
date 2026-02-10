/**
 * Complex number operations
 *
 * Pure functions for complex array operations:
 * real, imag, conj, angle
 *
 * @module ops/complex
 */

import { ArrayStorage } from '../storage';
import {
  isComplexDType,
  getComplexComponentDType,
  getTypedArrayConstructor,
  type DType,
} from '../dtype';
import { Complex } from '../complex';

/**
 * Return the real part of complex argument.
 *
 * For complex arrays, returns the real components.
 * For real arrays, returns a copy of the input.
 *
 * @param a - Input array storage
 * @returns Array with real parts (float64 or float32 for complex, same dtype for real)
 */
export function real(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const shape = Array.from(a.shape);
  const size = a.size;

  if (isComplexDType(dtype)) {
    // Complex array: extract real parts
    const resultDtype = getComplexComponentDType(dtype);
    const result = ArrayStorage.zeros(shape, resultDtype);
    const resultData = result.data;
    const srcData = a.data as Float64Array | Float32Array;

    // Data is interleaved [re, im, re, im, ...]
    for (let i = 0; i < size; i++) {
      (resultData as Float64Array | Float32Array)[i] = srcData[i * 2]!;
    }

    return result;
  }

  // Real array: return copy
  return a.copy();
}

/**
 * Return the imaginary part of complex argument.
 *
 * For complex arrays, returns the imaginary components.
 * For real arrays, returns zeros.
 *
 * @param a - Input array storage
 * @returns Array with imaginary parts
 */
export function imag(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const shape = Array.from(a.shape);
  const size = a.size;

  if (isComplexDType(dtype)) {
    // Complex array: extract imaginary parts
    const resultDtype = getComplexComponentDType(dtype);
    const result = ArrayStorage.zeros(shape, resultDtype);
    const resultData = result.data;
    const srcData = a.data as Float64Array | Float32Array;

    // Data is interleaved [re, im, re, im, ...]
    for (let i = 0; i < size; i++) {
      (resultData as Float64Array | Float32Array)[i] = srcData[i * 2 + 1]!;
    }

    return result;
  }

  // Real array: return zeros with appropriate dtype
  // NumPy returns float64 for integer types
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  return ArrayStorage.zeros(shape, resultDtype);
}

/**
 * Return the complex conjugate.
 *
 * For complex arrays, negates the imaginary part: (a + bi) -> (a - bi)
 * For real arrays, returns a copy of the input.
 *
 * @param a - Input array storage
 * @returns Complex conjugate array
 */
export function conj(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const shape = Array.from(a.shape);
  const size = a.size;

  if (isComplexDType(dtype)) {
    // Complex array: negate imaginary parts
    const Constructor = getTypedArrayConstructor(dtype)!;
    const physicalSize = size * 2;
    const resultData = new Constructor(physicalSize);
    const srcData = a.data as Float64Array | Float32Array;

    // Data is interleaved [re, im, re, im, ...]
    for (let i = 0; i < size; i++) {
      (resultData as Float64Array | Float32Array)[i * 2] = srcData[i * 2]!; // real stays same
      (resultData as Float64Array | Float32Array)[i * 2 + 1] = -srcData[i * 2 + 1]!; // negate imag
    }

    return ArrayStorage.fromData(resultData, shape, dtype);
  }

  // Real array: return copy
  return a.copy();
}

// Alias for conj
export const conjugate = conj;

/**
 * Return the angle (phase) of complex argument.
 *
 * angle(z) = arctan2(imag(z), real(z))
 *
 * For real arrays, returns 0 for positive, pi for negative.
 *
 * @param a - Input array storage
 * @param deg - Return angle in degrees if true (default: false, returns radians)
 * @returns Array with angles in radians (or degrees)
 */
export function angle(a: ArrayStorage, deg: boolean = false): ArrayStorage {
  const dtype = a.dtype as DType;
  const shape = Array.from(a.shape);
  const size = a.size;

  // Result is always float64
  const result = ArrayStorage.zeros(shape, 'float64');
  const resultData = result.data as Float64Array;

  if (isComplexDType(dtype)) {
    const srcData = a.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;
      let ang = Math.atan2(im, re);
      if (deg) {
        ang = (ang * 180) / Math.PI;
      }
      resultData[i] = ang;
    }
  } else {
    // Real array: angle is 0 for positive, pi for negative
    for (let i = 0; i < size; i++) {
      const val = a.iget(i);
      const numVal = val instanceof Complex ? val.re : Number(val);
      let ang = numVal >= 0 ? 0 : Math.PI;
      if (deg) {
        ang = (ang * 180) / Math.PI;
      }
      resultData[i] = ang;
    }
  }

  return result;
}

/**
 * Return the absolute value (magnitude) for complex arrays.
 *
 * For complex: |z| = sqrt(re² + im²)
 * For real: |x| = abs(x)
 *
 * This is a helper that can be called from arithmetic.absolute
 *
 * @param a - Input array storage
 * @returns Array with magnitudes
 */
export function complexAbs(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const shape = Array.from(a.shape);
  const size = a.size;

  // For complex, result dtype is the component dtype
  const resultDtype = isComplexDType(dtype) ? getComplexComponentDType(dtype) : dtype;

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    const srcData = a.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;
      (resultData as Float64Array | Float32Array)[i] = Math.sqrt(re * re + im * im);
    }
  }

  return result;
}
