/**
 * Hyperbolic operations
 *
 * Pure functions for element-wise hyperbolic operations:
 * sinh, cosh, tanh, arcsinh, arccosh, arctanh
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../storage';
import { elementwiseUnaryOp } from '../internal/compute';
import { isComplexDType, type DType } from '../dtype';

/**
 * Hyperbolic sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: sinh(a+bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
 *
 * @param a - Input array storage
 * @returns Result storage with sinh applied
 */
export function sinh(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;

      // sinh(a+bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
      dstData[i * 2] = Math.sinh(re) * Math.cos(im);
      dstData[i * 2 + 1] = Math.cosh(re) * Math.sin(im);
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.sinh, false);
}

/**
 * Hyperbolic cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: cosh(a+bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
 *
 * @param a - Input array storage
 * @returns Result storage with cosh applied
 */
export function cosh(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;

      // cosh(a+bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
      dstData[i * 2] = Math.cosh(re) * Math.cos(im);
      dstData[i * 2 + 1] = Math.sinh(re) * Math.sin(im);
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.cosh, false);
}

/**
 * Hyperbolic tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: tanh(z) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))
 *
 * @param a - Input array storage
 * @returns Result storage with tanh applied
 */
export function tanh(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;

      // tanh(z) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))
      const denom = Math.cosh(2 * re) + Math.cos(2 * im);
      dstData[i * 2] = Math.sinh(2 * re) / denom;
      dstData[i * 2 + 1] = Math.sin(2 * im) / denom;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.tanh, false);
}

/**
 * Inverse hyperbolic sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: arcsinh(z) = log(z + sqrt(z² + 1))
 *
 * @param a - Input array storage
 * @returns Result storage with arcsinh applied
 */
export function arcsinh(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const zRe = srcData[i * 2]!;
      const zIm = srcData[i * 2 + 1]!;

      // arcsinh(z) = log(z + sqrt(z² + 1))
      // z² = (zRe + zIm*i)² = zRe² - zIm² + 2*zRe*zIm*i
      const z2Re = zRe * zRe - zIm * zIm;
      const z2Im = 2 * zRe * zIm;

      // z² + 1
      const z2p1Re = z2Re + 1;
      const z2p1Im = z2Im;

      // sqrt(z² + 1)
      const mag = Math.sqrt(z2p1Re * z2p1Re + z2p1Im * z2p1Im);
      const sqrtRe = Math.sqrt((mag + z2p1Re) / 2);
      const sqrtIm = (z2p1Im >= 0 ? 1 : -1) * Math.sqrt((mag - z2p1Re) / 2);

      // z + sqrt(z² + 1)
      const sumRe = zRe + sqrtRe;
      const sumIm = zIm + sqrtIm;

      // log(...)
      const logMag = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
      dstData[i * 2] = Math.log(logMag);
      dstData[i * 2 + 1] = Math.atan2(sumIm, sumRe);
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.asinh, false);
}

/**
 * Inverse hyperbolic cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: arccosh(z) = log(z + sqrt(z - 1) * sqrt(z + 1))
 *
 * @param a - Input array storage (values >= 1)
 * @returns Result storage with arccosh applied
 */
export function arccosh(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const zRe = srcData[i * 2]!;
      const zIm = srcData[i * 2 + 1]!;

      // arccosh(z) = log(z + sqrt(z² - 1))
      // z² = zRe² - zIm² + 2*zRe*zIm*i
      const z2Re = zRe * zRe - zIm * zIm;
      const z2Im = 2 * zRe * zIm;

      // z² - 1
      const z2m1Re = z2Re - 1;
      const z2m1Im = z2Im;

      // sqrt(z² - 1)
      const mag = Math.sqrt(z2m1Re * z2m1Re + z2m1Im * z2m1Im);
      const sqrtRe = Math.sqrt((mag + z2m1Re) / 2);
      const sqrtIm = (z2m1Im >= 0 ? 1 : -1) * Math.sqrt((mag - z2m1Re) / 2);

      // z + sqrt(z² - 1)
      const sumRe = zRe + sqrtRe;
      const sumIm = zIm + sqrtIm;

      // log(...)
      const logMag = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
      let resultRe = Math.log(logMag);
      let resultIm = Math.atan2(sumIm, sumRe);

      // Branch cut adjustment: for purely real z < 1, we need to adjust
      // to match NumPy's branch cut convention
      if (Math.abs(zIm) < 1e-15 && zRe < 1) {
        resultIm = -resultIm;
      }

      dstData[i * 2] = resultRe;
      dstData[i * 2 + 1] = resultIm;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.acosh, false);
}

/**
 * Inverse hyperbolic tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: arctanh(z) = (1/2) * log((1+z)/(1-z))
 *
 * @param a - Input array storage (values in range (-1, 1))
 * @returns Result storage with arctanh applied
 */
export function arctanh(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const zRe = srcData[i * 2]!;
      const zIm = srcData[i * 2 + 1]!;

      // arctanh(z) = (1/2) * log((1+z)/(1-z))
      // 1 + z
      const numRe = 1 + zRe;
      const numIm = zIm;

      // 1 - z
      const denRe = 1 - zRe;
      const denIm = -zIm;

      // (1+z) / (1-z) = num / den
      const denMagSq = denRe * denRe + denIm * denIm;
      const divRe = (numRe * denRe + numIm * denIm) / denMagSq;
      const divIm = (numIm * denRe - numRe * denIm) / denMagSq;

      // log((1+z)/(1-z))
      const logMag = Math.sqrt(divRe * divRe + divIm * divIm);
      const logRe = Math.log(logMag);
      const logIm = Math.atan2(divIm, divRe);

      // (1/2) * log(...)
      dstData[i * 2] = logRe / 2;
      dstData[i * 2 + 1] = logIm / 2;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.atanh, false);
}
