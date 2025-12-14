/**
 * Exponential, logarithmic, and power operations
 *
 * Pure functions for element-wise exponential operations:
 * exp, exp2, expm1, log, log2, log10, log1p, logaddexp, logaddexp2, sqrt, power
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { elementwiseUnaryOp, elementwiseBinaryOp, broadcastShapes } from '../internal/compute';
import { isBigIntDType, isComplexDType, type DType } from '../core/dtype';

/**
 * Square root of each element
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: sqrt(a+bi) = sqrt((|z|+a)/2) + sign(b)*i*sqrt((|z|-a)/2)
 *
 * @param a - Input array storage
 * @returns Result storage with sqrt applied
 */
export function sqrt(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;

  if (isComplexDType(dtype)) {
    const shape = Array.from(a.shape);
    const size = a.size;
    const srcData = a.data as Float64Array | Float32Array;

    // Result is same complex type
    const result = ArrayStorage.zeros(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;

      // |z| = sqrt(re² + im²)
      const mag = Math.sqrt(re * re + im * im);

      // sqrt(a+bi) = sqrt((|z|+a)/2) + sign(b)*i*sqrt((|z|-a)/2)
      const realPart = Math.sqrt((mag + re) / 2);
      const imagPart = (im >= 0 ? 1 : -1) * Math.sqrt((mag - re) / 2);

      dstData[i * 2] = realPart;
      dstData[i * 2 + 1] = imagPart;
    }

    return result;
  }

  return elementwiseUnaryOp(a, Math.sqrt, false); // false = promote integers to float64
}

/**
 * Raise elements to power
 * NumPy behavior: Promotes to float64 for integer types with non-integer exponents
 * For complex: z^n = |z|^n * (cos(n*θ) + i*sin(n*θ)) where θ = atan2(im, re)
 *
 * @param a - Base array storage
 * @param b - Exponent (array storage or scalar)
 * @returns Result storage with power applied
 */
export function power(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return powerScalar(a, b);
  }

  // Check for complex types
  const aIsComplex = isComplexDType(a.dtype);
  const bIsComplex = isComplexDType(b.dtype);

  if (aIsComplex || bIsComplex) {
    return complexPowerArray(a, b);
  }

  return elementwiseBinaryOp(a, b, Math.pow, 'power');
}

/**
 * Complex power with array exponent
 * @private
 */
function complexPowerArray(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const aIsComplex = isComplexDType(a.dtype);
  const bIsComplex = isComplexDType(b.dtype);

  // Result dtype: complex128 if any is complex128 or float64, else complex64
  const resultDtype: DType =
    a.dtype === 'complex128' || b.dtype === 'complex128' || b.dtype === 'float64' ? 'complex128' : 'complex64';

  const shape = Array.from(a.shape);
  const size = a.size;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const dstData = result.data as Float64Array | Float32Array;

  for (let i = 0; i < size; i++) {
    // Get base as complex
    let baseRe: number, baseIm: number;
    if (aIsComplex) {
      const srcData = a.data as Float64Array | Float32Array;
      baseRe = srcData[i * 2]!;
      baseIm = srcData[i * 2 + 1]!;
    } else {
      baseRe = Number(a.iget(i));
      baseIm = 0;
    }

    // Get exponent as complex
    let expRe: number, expIm: number;
    if (bIsComplex) {
      const srcData = b.data as Float64Array | Float32Array;
      expRe = srcData[i * 2]!;
      expIm = srcData[i * 2 + 1]!;
    } else {
      expRe = Number(b.iget(i));
      expIm = 0;
    }

    // z^w = exp(w * ln(z))
    // ln(z) = ln(|z|) + i*arg(z)
    const mag = Math.sqrt(baseRe * baseRe + baseIm * baseIm);
    const arg = Math.atan2(baseIm, baseRe);
    const lnMag = Math.log(mag);

    // w * ln(z) = (expRe + expIm*i) * (lnMag + arg*i)
    //           = (expRe*lnMag - expIm*arg) + (expRe*arg + expIm*lnMag)*i
    const realPart = expRe * lnMag - expIm * arg;
    const imagPart = expRe * arg + expIm * lnMag;

    // exp(realPart + imagPart*i) = exp(realPart) * (cos(imagPart) + i*sin(imagPart))
    const expReal = Math.exp(realPart);
    dstData[i * 2] = expReal * Math.cos(imagPart);
    dstData[i * 2 + 1] = expReal * Math.sin(imagPart);
  }

  return result;
}

/**
 * Power with scalar exponent (optimized path)
 * @private
 */
function powerScalar(storage: ArrayStorage, exponent: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Handle complex types
  if (isComplexDType(dtype)) {
    // z^n = |z|^n * (cos(n*θ) + i*sin(n*θ))
    const result = ArrayStorage.zeros(shape, dtype);
    const srcData = data as Float64Array | Float32Array;
    const dstData = result.data as Float64Array | Float32Array;

    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;

      const mag = Math.sqrt(re * re + im * im);
      const arg = Math.atan2(im, re);

      const newMag = Math.pow(mag, exponent);
      const newArg = arg * exponent;

      dstData[i * 2] = newMag * Math.cos(newArg);
      dstData[i * 2 + 1] = newMag * Math.sin(newArg);
    }

    return result;
  }

  // NumPy behavior: integer ** integer stays integer if exponent >= 0
  // integer ** negative or float exponent promotes to float64
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const needsFloatPromotion = isIntegerType && (exponent < 0 || !Number.isInteger(exponent));
  const resultDtype = needsFloatPromotion ? 'float64' : dtype;

  // Create result with appropriate dtype
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    if (isBigIntDType(resultDtype) && Number.isInteger(exponent) && exponent >= 0) {
      // BigInt ** positive integer stays BigInt
      const thisTyped = data as BigInt64Array | BigUint64Array;
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = thisTyped[i]! ** BigInt(exponent);
      }
    } else {
      // BigInt ** negative or float promotes to float64
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.pow(Number(data[i]!), exponent);
      }
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.pow(Number(data[i]!), exponent);
    }
  }

  return result;
}

/**
 * Natural exponential function (e^x) for each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with exp applied
 */
export function exp(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.exp, false);
}

/**
 * Base-2 exponential function (2^x) for each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with 2^x applied
 */
export function exp2(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, (x) => Math.pow(2, x), false);
}

/**
 * Exponential minus one (e^x - 1) for each element
 * More accurate than exp(x) - 1 for small x
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with expm1 applied
 */
export function expm1(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.expm1, false);
}

/**
 * Natural logarithm (ln) for each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with log applied
 */
export function log(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.log, false);
}

/**
 * Base-2 logarithm for each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with log2 applied
 */
export function log2(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.log2, false);
}

/**
 * Base-10 logarithm for each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with log10 applied
 */
export function log10(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.log10, false);
}

/**
 * Natural logarithm of (1 + x) for each element
 * More accurate than log(1 + x) for small x
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with log1p applied
 */
export function log1p(a: ArrayStorage): ArrayStorage {
  return elementwiseUnaryOp(a, Math.log1p, false);
}

/**
 * Logarithm of the sum of exponentials: log(exp(x1) + exp(x2))
 * More numerically stable than computing the expression directly
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param x1 - First input array storage
 * @param x2 - Second input array storage (or scalar)
 * @returns Result storage with logaddexp applied
 */
export function logaddexp(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  if (typeof x2 === 'number') {
    return logaddexpScalar(x1, x2);
  }
  return logaddexpArray(x1, x2);
}

/**
 * logaddexp with array x2 (always returns float64 for non-float32 inputs)
 * @private
 */
function logaddexpArray(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const outputShape = broadcastShapes(x1.shape, x2.shape);
  const size = outputShape.reduce((a, b) => a * b, 1);
  const dtype1 = x1.dtype;
  const dtype2 = x2.dtype;

  // Always promote to float64 for logaddexp (matching NumPy behavior)
  // Only preserve float32 if both inputs are float32
  const resultDtype = dtype1 === 'float32' && dtype2 === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(outputShape, resultDtype);
  const resultData = result.data;

  // Use broadcast iteration
  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype1) ? Number(x1.iget(i)) : Number(x1.iget(i));
    const val2 = isBigIntDType(dtype2) ? Number(x2.iget(i)) : Number(x2.iget(i));

    // Numerically stable implementation: log(exp(a) + exp(b)) = max(a,b) + log1p(exp(-|a-b|))
    const maxVal = Math.max(val1, val2);
    const minVal = Math.min(val1, val2);
    resultData[i] = maxVal + Math.log1p(Math.exp(minVal - maxVal));
  }

  return result;
}

/**
 * logaddexp with scalar x2 (optimized path)
 * @private
 */
function logaddexpScalar(storage: ArrayStorage, x2: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const size = storage.size;

  // Always promote to float64 for logaddexp
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype) ? Number(storage.data[i]!) : Number(storage.data[i]!);

    // Numerically stable implementation
    const maxVal = Math.max(val1, x2);
    const minVal = Math.min(val1, x2);
    resultData[i] = maxVal + Math.log1p(Math.exp(minVal - maxVal));
  }

  return result;
}

/**
 * Logarithm base 2 of the sum of exponentials: log2(2^x1 + 2^x2)
 * More numerically stable than computing the expression directly
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param x1 - First input array storage
 * @param x2 - Second input array storage (or scalar)
 * @returns Result storage with logaddexp2 applied
 */
export function logaddexp2(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  if (typeof x2 === 'number') {
    return logaddexp2Scalar(x1, x2);
  }
  return logaddexp2Array(x1, x2);
}

/**
 * logaddexp2 with array x2 (always returns float64 for non-float32 inputs)
 * @private
 */
function logaddexp2Array(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const outputShape = broadcastShapes(x1.shape, x2.shape);
  const size = outputShape.reduce((a, b) => a * b, 1);
  const dtype1 = x1.dtype;
  const dtype2 = x2.dtype;

  // Always promote to float64 for logaddexp2 (matching NumPy behavior)
  // Only preserve float32 if both inputs are float32
  const resultDtype = dtype1 === 'float32' && dtype2 === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(outputShape, resultDtype);
  const resultData = result.data;

  const LOG2_E = Math.LOG2E; // log2(e)

  // Use broadcast iteration
  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype1) ? Number(x1.iget(i)) : Number(x1.iget(i));
    const val2 = isBigIntDType(dtype2) ? Number(x2.iget(i)) : Number(x2.iget(i));

    // Numerically stable implementation: log2(2^a + 2^b) = max(a,b) + log2(1 + 2^(-|a-b|))
    const maxVal = Math.max(val1, val2);
    const minVal = Math.min(val1, val2);
    // log2(1 + x) = log(1 + x) * log2(e)
    resultData[i] = maxVal + Math.log1p(Math.pow(2, minVal - maxVal)) * LOG2_E;
  }

  return result;
}

/**
 * logaddexp2 with scalar x2 (optimized path)
 * @private
 */
function logaddexp2Scalar(storage: ArrayStorage, x2: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const size = storage.size;

  // Always promote to float64 for logaddexp2
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  const LOG2_E = Math.LOG2E;

  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype) ? Number(storage.data[i]!) : Number(storage.data[i]!);

    // Numerically stable implementation
    const maxVal = Math.max(val1, x2);
    const minVal = Math.min(val1, x2);
    resultData[i] = maxVal + Math.log1p(Math.pow(2, minVal - maxVal)) * LOG2_E;
  }

  return result;
}
