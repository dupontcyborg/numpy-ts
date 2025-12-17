/**
 * Trigonometric operations
 *
 * Pure functions for element-wise trigonometric operations:
 * sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot, degrees, radians
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { elementwiseUnaryOp } from '../internal/compute';
import { isBigIntDType, isComplexDType, throwIfComplex, type DType } from '../core/dtype';

/**
 * Sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
 *
 * @param a - Input array storage (angles in radians)
 * @returns Result storage with sin applied
 */
export function sin(a: ArrayStorage): ArrayStorage {
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

      // sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
      dstData[i * 2] = Math.sin(re) * Math.cosh(im);
      dstData[i * 2 + 1] = Math.cos(re) * Math.sinh(im);
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.sin, false);
}

/**
 * Cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
 *
 * @param a - Input array storage (angles in radians)
 * @returns Result storage with cos applied
 */
export function cos(a: ArrayStorage): ArrayStorage {
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

      // cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
      dstData[i * 2] = Math.cos(re) * Math.cosh(im);
      dstData[i * 2 + 1] = -Math.sin(re) * Math.sinh(im);
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.cos, false);
}

/**
 * Tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: tan(a+bi) = (sin(2a) + i*sinh(2b)) / (cos(2a) + cosh(2b))
 *
 * @param a - Input array storage (angles in radians)
 * @returns Result storage with tan applied
 */
export function tan(a: ArrayStorage): ArrayStorage {
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

      // tan(a+bi) = (sin(2a) + i*sinh(2b)) / (cos(2a) + cosh(2b))
      const denom = Math.cos(2 * re) + Math.cosh(2 * im);
      dstData[i * 2] = Math.sin(2 * re) / denom;
      dstData[i * 2 + 1] = Math.sinh(2 * im) / denom;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.tan, false);
}

/**
 * Inverse sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: arcsin(z) = -i * log(iz + sqrt(1 - z²))
 *
 * @param a - Input array storage (values in range [-1, 1])
 * @returns Result storage with arcsin applied (radians)
 */
export function arcsin(a: ArrayStorage): ArrayStorage {
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

      // arcsin(z) = -i * log(iz + sqrt(1 - z²))
      // iz = i*(zRe + zIm*i) = -zIm + zRe*i
      const izRe = -zIm;
      const izIm = zRe;

      // z² = (zRe + zIm*i)² = zRe² - zIm² + 2*zRe*zIm*i
      const z2Re = zRe * zRe - zIm * zIm;
      const z2Im = 2 * zRe * zIm;

      // 1 - z² = (1 - z2Re) - z2Im*i
      const oneMinusZ2Re = 1 - z2Re;
      const oneMinusZ2Im = -z2Im;

      // sqrt(1 - z²)
      const mag = Math.sqrt(oneMinusZ2Re * oneMinusZ2Re + oneMinusZ2Im * oneMinusZ2Im);
      const sqrtRe = Math.sqrt((mag + oneMinusZ2Re) / 2);
      const sqrtIm = (oneMinusZ2Im >= 0 ? 1 : -1) * Math.sqrt((mag - oneMinusZ2Re) / 2);

      // iz + sqrt(1 - z²)
      const sumRe = izRe + sqrtRe;
      const sumIm = izIm + sqrtIm;

      // log(...)
      const logMag = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
      const logRe = Math.log(logMag);
      const logIm = Math.atan2(sumIm, sumRe);

      // -i * log = -i * (logRe + logIm*i) = logIm - logRe*i
      let resultRe = logIm;
      let resultIm = -logRe;

      // Branch cut adjustment: for purely real z > 1, the standard formula gives
      // the wrong sign for the imaginary part. NumPy defines arcsin(x) for x > 1
      // as pi/2 + i*arccosh(x), but our formula gives pi/2 - i*arccosh(x).
      // Fix: negate imaginary part when z is real and z > 1
      if (Math.abs(zIm) < 1e-15 && zRe > 1) {
        resultIm = -resultIm;
      }

      dstData[i * 2] = resultRe;
      dstData[i * 2 + 1] = resultIm;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.asin, false);
}

/**
 * Inverse cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: arccos(z) = -i * log(z + i*sqrt(1 - z²))
 *
 * @param a - Input array storage (values in range [-1, 1])
 * @returns Result storage with arccos applied (radians)
 */
export function arccos(a: ArrayStorage): ArrayStorage {
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

      // arccos(z) = -i * log(z + i*sqrt(1 - z²))
      // z² = (zRe + zIm*i)² = zRe² - zIm² + 2*zRe*zIm*i
      const z2Re = zRe * zRe - zIm * zIm;
      const z2Im = 2 * zRe * zIm;

      // 1 - z² = (1 - z2Re) - z2Im*i
      const oneMinusZ2Re = 1 - z2Re;
      const oneMinusZ2Im = -z2Im;

      // sqrt(1 - z²)
      const mag = Math.sqrt(oneMinusZ2Re * oneMinusZ2Re + oneMinusZ2Im * oneMinusZ2Im);
      const sqrtRe = Math.sqrt((mag + oneMinusZ2Re) / 2);
      const sqrtIm = (oneMinusZ2Im >= 0 ? 1 : -1) * Math.sqrt((mag - oneMinusZ2Re) / 2);

      // i * sqrt(1 - z²) = i*(sqrtRe + sqrtIm*i) = -sqrtIm + sqrtRe*i
      const iSqrtRe = -sqrtIm;
      const iSqrtIm = sqrtRe;

      // z + i*sqrt(1 - z²)
      const sumRe = zRe + iSqrtRe;
      const sumIm = zIm + iSqrtIm;

      // log(...)
      const logMag = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
      const logRe = Math.log(logMag);
      const logIm = Math.atan2(sumIm, sumRe);

      // -i * log = -i * (logRe + logIm*i) = logIm - logRe*i
      let resultRe = logIm;
      let resultIm = -logRe;

      // Branch cut adjustment: for purely real z > 1, the standard formula gives
      // the wrong sign for the imaginary part. NumPy defines arccos(x) for x > 1
      // as -i*arccosh(x), but our formula gives +i*arccosh(x).
      // Fix: negate imaginary part when z is real and z > 1
      if (Math.abs(zIm) < 1e-15 && zRe > 1) {
        resultIm = -resultIm;
      }

      dstData[i * 2] = resultRe;
      dstData[i * 2 + 1] = resultIm;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.acos, false);
}

/**
 * Inverse tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: arctan(z) = (i/2) * log((1 - iz) / (1 + iz))
 *
 * @param a - Input array storage
 * @returns Result storage with arctan applied (radians)
 */
export function arctan(a: ArrayStorage): ArrayStorage {
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

      // arctan(z) = (i/2) * log((1 - iz) / (1 + iz))
      // iz = -zIm + zRe*i
      const izRe = -zIm;
      const izIm = zRe;

      // 1 - iz
      const numRe = 1 - izRe;
      const numIm = -izIm;

      // 1 + iz
      const denRe = 1 + izRe;
      const denIm = izIm;

      // (1 - iz) / (1 + iz) - complex division
      const denMag2 = denRe * denRe + denIm * denIm;
      const divRe = (numRe * denRe + numIm * denIm) / denMag2;
      const divIm = (numIm * denRe - numRe * denIm) / denMag2;

      // log(...)
      const logMag = Math.sqrt(divRe * divRe + divIm * divIm);
      const logRe = Math.log(logMag);
      const logIm = Math.atan2(divIm, divRe);

      // (i/2) * log = (i/2) * (logRe + logIm*i) = -logIm/2 + logRe/2 * i
      dstData[i * 2] = -logIm / 2;
      dstData[i * 2 + 1] = logRe / 2;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.atan, false);
}

/**
 * Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param x1 - y-coordinates
 * @param x2 - x-coordinates (array storage or scalar)
 * @returns Angle in radians between -π and π
 */
export function arctan2(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'arctan2', 'arctan2 is only defined for real numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'arctan2', 'arctan2 is only defined for real numbers.');
  }
  if (typeof x2 === 'number') {
    return arctan2Scalar(x1, x2);
  }
  return arctan2Array(x1, x2);
}

/**
 * arctan2 with array x2 (always returns float64 for non-float32 inputs)
 * @private
 */
function arctan2Array(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const shape = Array.from(x1.shape);
  const size = x1.size;
  const dtype1 = x1.dtype;
  const dtype2 = x2.dtype;

  // Always promote to float64 for arctan2 (matching NumPy behavior)
  // Only preserve float32 if both inputs are float32
  const resultDtype = dtype1 === 'float32' && dtype2 === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype1) ? Number(x1.data[i]!) : Number(x1.data[i]!);
    const val2 = isBigIntDType(dtype2) ? Number(x2.data[i]!) : Number(x2.data[i]!);
    resultData[i] = Math.atan2(val1, val2);
  }

  return result;
}

/**
 * arctan2 with scalar x2 (optimized path)
 * @private
 */
function arctan2Scalar(storage: ArrayStorage, x2: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Always promote to float64 for trig operations
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.atan2(Number(data[i]!), x2);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.atan2(Number(data[i]!), x2);
    }
  }

  return result;
}

/**
 * Given the "legs" of a right triangle, return its hypotenuse.
 * Equivalent to sqrt(x1**2 + x2**2), element-wise.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param x1 - First leg
 * @param x2 - Second leg (array storage or scalar)
 * @returns Hypotenuse
 */
export function hypot(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'hypot', 'hypot is only defined for real numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'hypot', 'hypot is only defined for real numbers.');
  }
  if (typeof x2 === 'number') {
    return hypotScalar(x1, x2);
  }
  return hypotArray(x1, x2);
}

/**
 * hypot with array x2 (always returns float64 for non-float32 inputs)
 * @private
 */
function hypotArray(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const shape = Array.from(x1.shape);
  const size = x1.size;
  const dtype1 = x1.dtype;
  const dtype2 = x2.dtype;

  // Always promote to float64 for hypot (matching NumPy behavior)
  // Only preserve float32 if both inputs are float32
  const resultDtype = dtype1 === 'float32' && dtype2 === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype1) ? Number(x1.data[i]!) : Number(x1.data[i]!);
    const val2 = isBigIntDType(dtype2) ? Number(x2.data[i]!) : Number(x2.data[i]!);
    resultData[i] = Math.hypot(val1, val2);
  }

  return result;
}

/**
 * hypot with scalar x2 (optimized path)
 * @private
 */
function hypotScalar(storage: ArrayStorage, x2: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Always promote to float64 for trig operations
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.hypot(Number(data[i]!), x2);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.hypot(Number(data[i]!), x2);
    }
  }

  return result;
}

/**
 * Convert angles from radians to degrees.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Angles in degrees
 */
export function degrees(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'degrees', 'degrees is only defined for real numbers.');
  const factor = 180 / Math.PI;
  return elementwiseUnaryOp(a, (x) => x * factor, false);
}

/**
 * Convert angles from degrees to radians.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in degrees)
 * @returns Angles in radians
 */
export function radians(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'radians', 'radians is only defined for real numbers.');
  const factor = Math.PI / 180;
  return elementwiseUnaryOp(a, (x) => x * factor, false);
}

/**
 * Convert angles from degrees to radians (alias for radians).
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in degrees)
 * @returns Angles in radians
 */
export function deg2rad(a: ArrayStorage): ArrayStorage {
  return radians(a);
}

/**
 * Convert angles from radians to degrees (alias for degrees).
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Angles in degrees
 */
export function rad2deg(a: ArrayStorage): ArrayStorage {
  return degrees(a);
}
