/**
 * Logic operations
 *
 * Pure functions for element-wise logical operations:
 * logical_and, logical_or, logical_not, logical_xor,
 * isfinite, isinf, isnan, isnat, copysign, signbit, nextafter, spacing
 *
 * These operations convert values to boolean (non-zero = true, zero = false)
 * and return boolean arrays (dtype: 'bool').
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType, isComplexDType, throwIfComplex, type DType } from '../core/dtype';
import { elementwiseComparisonOp } from '../internal/compute';
import { broadcastShapes } from '../internal/compute';

/**
 * Helper: Convert value to boolean (0 = false, non-zero = true)
 */
function toBool(val: number | bigint): boolean {
  return val !== 0 && val !== 0n;
}

/**
 * Helper: Check if a complex value at index i is truthy (non-zero)
 * For complex arrays, a value is truthy if either real or imag part is non-zero
 */
function isComplexTruthy(data: Float64Array | Float32Array, index: number): boolean {
  const re = data[index * 2]!;
  const im = data[index * 2 + 1]!;
  return re !== 0 || im !== 0;
}

/**
 * Helper: Check if two arrays can use the fast path
 * (both C-contiguous with same shape, no broadcasting needed)
 */
function canUseFastPath(a: ArrayStorage, b: ArrayStorage): boolean {
  return (
    a.isCContiguous &&
    b.isCContiguous &&
    a.shape.length === b.shape.length &&
    a.shape.every((dim, i) => dim === b.shape[i])
  );
}

// ============================================================
// Logical Binary Operations
// ============================================================

/**
 * Logical AND of two arrays or array and scalar
 *
 * Returns a boolean array where each element is the logical AND
 * of corresponding elements (non-zero = true, zero = false).
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Boolean result storage
 */
export function logical_and(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return logicalAndScalar(a, b);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return logicalAndArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseComparisonOp(a, b, (x, y) => toBool(x) && toBool(y));
}

/**
 * Fast path for logical AND of two contiguous arrays
 * @private
 */
function logicalAndArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const aData = a.data;
  const bData = b.data;
  const size = a.size;

  const aIsBigInt = isBigIntDType(a.dtype);
  const bIsBigInt = isBigIntDType(b.dtype);
  const aIsComplex = isComplexDType(a.dtype);
  const bIsComplex = isComplexDType(b.dtype);

  if (aIsComplex || bIsComplex) {
    for (let i = 0; i < size; i++) {
      const aVal = aIsComplex
        ? isComplexTruthy(aData as Float64Array | Float32Array, i)
        : (aData[i] as number) !== 0;
      const bVal = bIsComplex
        ? isComplexTruthy(bData as Float64Array | Float32Array, i)
        : (bData[i] as number) !== 0;
      data[i] = aVal && bVal ? 1 : 0;
    }
  } else if (aIsBigInt || bIsBigInt) {
    for (let i = 0; i < size; i++) {
      const aVal = aIsBigInt ? (aData[i] as bigint) !== 0n : (aData[i] as number) !== 0;
      const bVal = bIsBigInt ? (bData[i] as bigint) !== 0n : (bData[i] as number) !== 0;
      data[i] = aVal && bVal ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = (aData[i] as number) !== 0 && (bData[i] as number) !== 0 ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Logical AND with scalar (optimized path)
 * @private
 */
function logicalAndScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const scalarBool = scalar !== 0;
  const size = storage.size;

  if (isComplexDType(storage.dtype)) {
    const typedData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      data[i] = isComplexTruthy(typedData, i) && scalarBool ? 1 : 0;
    }
  } else if (isBigIntDType(storage.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      data[i] = typedData[i] !== 0n && scalarBool ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = (thisData[i] as number) !== 0 && scalarBool ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

/**
 * Logical OR of two arrays or array and scalar
 *
 * Returns a boolean array where each element is the logical OR
 * of corresponding elements (non-zero = true, zero = false).
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Boolean result storage
 */
export function logical_or(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return logicalOrScalar(a, b);
  }

  // Optimize single-element non-complex arrays as scalars (only when same dtype)
  if (b.size === 1 && !isComplexDType(b.dtype) && a.dtype === b.dtype) {
    const scalarVal = Number(b.data[0]!);
    return logicalOrScalar(a, scalarVal);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return logicalOrArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseComparisonOp(a, b, (x, y) => toBool(x) || toBool(y));
}

/**
 * Fast path for logical OR of two contiguous arrays
 * @private
 */
function logicalOrArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const aData = a.data;
  const bData = b.data;
  const size = a.size;

  const aIsBigInt = isBigIntDType(a.dtype);
  const bIsBigInt = isBigIntDType(b.dtype);
  const aIsComplex = isComplexDType(a.dtype);
  const bIsComplex = isComplexDType(b.dtype);

  if (aIsComplex || bIsComplex) {
    for (let i = 0; i < size; i++) {
      const aVal = aIsComplex
        ? isComplexTruthy(aData as Float64Array | Float32Array, i)
        : (aData[i] as number) !== 0;
      const bVal = bIsComplex
        ? isComplexTruthy(bData as Float64Array | Float32Array, i)
        : (bData[i] as number) !== 0;
      data[i] = aVal || bVal ? 1 : 0;
    }
  } else if (aIsBigInt || bIsBigInt) {
    for (let i = 0; i < size; i++) {
      const aVal = aIsBigInt ? (aData[i] as bigint) !== 0n : (aData[i] as number) !== 0;
      const bVal = bIsBigInt ? (bData[i] as bigint) !== 0n : (bData[i] as number) !== 0;
      data[i] = aVal || bVal ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = (aData[i] as number) !== 0 || (bData[i] as number) !== 0 ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Logical OR with scalar (optimized path)
 * @private
 */
function logicalOrScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const scalarBool = scalar !== 0;
  const size = storage.size;

  if (isComplexDType(storage.dtype)) {
    const typedData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      data[i] = isComplexTruthy(typedData, i) || scalarBool ? 1 : 0;
    }
  } else if (isBigIntDType(storage.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      data[i] = typedData[i] !== 0n || scalarBool ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = (thisData[i] as number) !== 0 || scalarBool ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

/**
 * Logical NOT of an array
 *
 * Returns a boolean array where each element is the logical NOT
 * of the input (non-zero = false, zero = true).
 *
 * @param a - Input array storage
 * @returns Boolean result storage
 */
export function logical_not(a: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isComplexDType(a.dtype)) {
    const typedData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      // NOT of complex: true if both real and imag are zero
      data[i] = !isComplexTruthy(typedData, i) ? 1 : 0;
    }
  } else if (isBigIntDType(a.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      data[i] = typedData[i] === 0n ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = (thisData[i] as number) === 0 ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Logical XOR of two arrays or array and scalar
 *
 * Returns a boolean array where each element is the logical XOR
 * of corresponding elements (non-zero = true, zero = false).
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Boolean result storage
 */
export function logical_xor(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return logicalXorScalar(a, b);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return logicalXorArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseComparisonOp(a, b, (x, y) => toBool(x) !== toBool(y));
}

/**
 * Fast path for logical XOR of two contiguous arrays
 * @private
 */
function logicalXorArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const aData = a.data;
  const bData = b.data;
  const size = a.size;

  const aIsBigInt = isBigIntDType(a.dtype);
  const bIsBigInt = isBigIntDType(b.dtype);
  const aIsComplex = isComplexDType(a.dtype);
  const bIsComplex = isComplexDType(b.dtype);

  if (aIsComplex || bIsComplex) {
    for (let i = 0; i < size; i++) {
      const aVal = aIsComplex
        ? isComplexTruthy(aData as Float64Array | Float32Array, i)
        : (aData[i] as number) !== 0;
      const bVal = bIsComplex
        ? isComplexTruthy(bData as Float64Array | Float32Array, i)
        : (bData[i] as number) !== 0;
      data[i] = aVal !== bVal ? 1 : 0;
    }
  } else if (aIsBigInt || bIsBigInt) {
    for (let i = 0; i < size; i++) {
      const aVal = aIsBigInt ? (aData[i] as bigint) !== 0n : (aData[i] as number) !== 0;
      const bVal = bIsBigInt ? (bData[i] as bigint) !== 0n : (bData[i] as number) !== 0;
      data[i] = aVal !== bVal ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const aVal = (aData[i] as number) !== 0;
      const bVal = (bData[i] as number) !== 0;
      data[i] = aVal !== bVal ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Logical XOR with scalar (optimized path)
 * @private
 */
function logicalXorScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const scalarBool = scalar !== 0;
  const size = storage.size;

  if (isComplexDType(storage.dtype)) {
    const typedData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const valBool = isComplexTruthy(typedData, i);
      data[i] = valBool !== scalarBool ? 1 : 0;
    }
  } else if (isBigIntDType(storage.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const valBool = typedData[i] !== 0n;
      data[i] = valBool !== scalarBool ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const valBool = (thisData[i] as number) !== 0;
      data[i] = valBool !== scalarBool ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

// ============================================================
// Floating-point Testing Functions
// ============================================================

/**
 * Test element-wise for finiteness (not infinity and not NaN)
 *
 * For complex numbers: True if both real and imaginary parts are finite.
 *
 * @param a - Input array storage
 * @returns Boolean result storage
 */
export function isfinite(a: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isComplexDType(a.dtype as DType)) {
    // Complex: finite if both real and imag are finite
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const re = complexData[i * 2]!;
      const im = complexData[i * 2 + 1]!;
      data[i] = Number.isFinite(re) && Number.isFinite(im) ? 1 : 0;
    }
  } else if (isBigIntDType(a.dtype)) {
    // BigInt values are always finite
    for (let i = 0; i < size; i++) {
      data[i] = 1;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = thisData[i] as number;
      data[i] = Number.isFinite(val) ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Test element-wise for positive or negative infinity
 *
 * For complex numbers: True if either real or imaginary part is infinite.
 *
 * @param a - Input array storage
 * @returns Boolean result storage
 */
export function isinf(a: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isComplexDType(a.dtype as DType)) {
    // Complex: infinite if either part is infinite
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const re = complexData[i * 2]!;
      const im = complexData[i * 2 + 1]!;
      const reInf = !Number.isFinite(re) && !Number.isNaN(re);
      const imInf = !Number.isFinite(im) && !Number.isNaN(im);
      data[i] = reInf || imInf ? 1 : 0;
    }
  } else if (isBigIntDType(a.dtype)) {
    // BigInt values are never infinite
    for (let i = 0; i < size; i++) {
      data[i] = 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = thisData[i] as number;
      data[i] = !Number.isFinite(val) && !Number.isNaN(val) ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Test element-wise for NaN (Not a Number)
 *
 * For complex numbers: True if either real or imaginary part is NaN.
 *
 * @param a - Input array storage
 * @returns Boolean result storage
 */
export function isnan(a: ArrayStorage): ArrayStorage {
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isComplexDType(a.dtype as DType)) {
    // Complex: NaN if either part is NaN
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const re = complexData[i * 2]!;
      const im = complexData[i * 2 + 1]!;
      data[i] = Number.isNaN(re) || Number.isNaN(im) ? 1 : 0;
    }
  } else if (isBigIntDType(a.dtype)) {
    // BigInt values are never NaN
    for (let i = 0; i < size; i++) {
      data[i] = 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = Number.isNaN(thisData[i] as number) ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Test element-wise for NaT (Not a Time)
 *
 * In NumPy, NaT is the datetime equivalent of NaN.
 * Since we don't have datetime support, this always returns false.
 *
 * @param a - Input array storage
 * @returns Boolean result storage (all false)
 */
export function isnat(a: ArrayStorage): ArrayStorage {
  // Without datetime support, nothing is NaT
  const data = new Uint8Array(a.size);
  // All zeros (false) by default
  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

// ============================================================
// Floating-point Manipulation Functions
// ============================================================

/**
 * Change the sign of x1 to that of x2, element-wise
 *
 * Returns a value with the magnitude of x1 and the sign of x2.
 *
 * @param x1 - Values to change sign of (magnitude source)
 * @param x2 - Values whose sign is used (sign source)
 * @returns Array with magnitude from x1 and sign from x2
 */
export function copysign(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'copysign', 'copysign is only defined for real numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'copysign', 'copysign is only defined for real numbers.');
  }
  if (typeof x2 === 'number') {
    return copysignScalar(x1, x2);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(x1, x2)) {
    return copysignArraysFast(x1, x2);
  }

  // Slow path: broadcasting
  const outputShape = broadcastShapes(x1.shape, x2.shape);
  const size = outputShape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  // Create broadcast views using the helper function
  const x1Broadcast = broadcastToStorage(x1, outputShape);
  const x2Broadcast = broadcastToStorage(x2, outputShape);

  for (let i = 0; i < size; i++) {
    const val1 = Number(x1Broadcast.iget(i));
    const val2 = Number(x2Broadcast.iget(i));
    resultData[i] = Math.sign(val2) * Math.abs(val1);
  }

  return result;
}

/**
 * Fast path for copysign of two contiguous arrays
 * @private
 */
function copysignArraysFast(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const result = ArrayStorage.zeros(Array.from(x1.shape), 'float64');
  const resultData = result.data as Float64Array;
  const size = x1.size;
  const x1Data = x1.data;
  const x2Data = x2.data;

  const x1IsBigInt = isBigIntDType(x1.dtype);
  const x2IsBigInt = isBigIntDType(x2.dtype);

  for (let i = 0; i < size; i++) {
    const val1 = x1IsBigInt ? Number(x1Data[i] as bigint) : (x1Data[i] as number);
    const val2 = x2IsBigInt ? Number(x2Data[i] as bigint) : (x2Data[i] as number);
    resultData[i] = Math.sign(val2) * Math.abs(val1);
  }

  return result;
}

/**
 * Copysign with scalar (optimized path)
 * @private
 */
function copysignScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.zeros(Array.from(storage.shape), 'float64');
  const resultData = result.data as Float64Array;
  const thisData = storage.data;
  const size = storage.size;
  const signVal = Math.sign(scalar);

  if (isBigIntDType(storage.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      resultData[i] = signVal * Math.abs(Number(typedData[i]));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = signVal * Math.abs(thisData[i] as number);
    }
  }

  return result;
}

/**
 * Returns element-wise True where signbit is set (less than zero)
 *
 * @param a - Input array storage
 * @returns Boolean result storage
 */
export function signbit(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'signbit', 'signbit is only defined for real numbers.');
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isBigIntDType(a.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      data[i] = typedData[i]! < 0n ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = thisData[i] as number;
      // Handle -0.0 case: Object.is distinguishes -0 from +0
      data[i] = val < 0 || Object.is(val, -0) ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Return the next floating-point value after x1 towards x2, element-wise
 *
 * @param x1 - Values to find the next representable value of
 * @param x2 - Direction to look in for the next representable value
 * @returns Array of next representable values
 */
export function nextafter(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'nextafter', 'nextafter is only defined for real numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'nextafter', 'nextafter is only defined for real numbers.');
  }
  if (typeof x2 === 'number') {
    return nextafterScalar(x1, x2);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(x1, x2)) {
    return nextafterArraysFast(x1, x2);
  }

  // Slow path: broadcasting
  const outputShape = broadcastShapes(x1.shape, x2.shape);
  const size = outputShape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  // Create broadcast views
  const x1Broadcast = broadcastToStorage(x1, outputShape);
  const x2Broadcast = broadcastToStorage(x2, outputShape);

  for (let i = 0; i < size; i++) {
    const val1 = Number(x1Broadcast.iget(i));
    const val2 = Number(x2Broadcast.iget(i));
    resultData[i] = nextafterSingle(val1, val2);
  }

  return result;
}

/**
 * Fast path for nextafter of two contiguous arrays
 * @private
 */
function nextafterArraysFast(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const result = ArrayStorage.zeros(Array.from(x1.shape), 'float64');
  const resultData = result.data as Float64Array;
  const size = x1.size;
  const x1Data = x1.data;
  const x2Data = x2.data;

  const x1IsBigInt = isBigIntDType(x1.dtype);
  const x2IsBigInt = isBigIntDType(x2.dtype);

  for (let i = 0; i < size; i++) {
    const val1 = x1IsBigInt ? Number(x1Data[i] as bigint) : (x1Data[i] as number);
    const val2 = x2IsBigInt ? Number(x2Data[i] as bigint) : (x2Data[i] as number);
    resultData[i] = nextafterSingle(val1, val2);
  }

  return result;
}

/**
 * Nextafter with scalar (optimized path)
 * @private
 */
function nextafterScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.zeros(Array.from(storage.shape), 'float64');
  const resultData = result.data as Float64Array;
  const thisData = storage.data;
  const size = storage.size;

  if (isBigIntDType(storage.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      resultData[i] = nextafterSingle(Number(typedData[i]), scalar);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = nextafterSingle(thisData[i] as number, scalar);
    }
  }

  return result;
}

/**
 * Compute nextafter for a single pair of values
 * @private
 */
function nextafterSingle(x: number, y: number): number {
  // Handle NaN
  if (Number.isNaN(x) || Number.isNaN(y)) {
    return NaN;
  }

  // Handle equality
  if (x === y) {
    return y;
  }

  // Handle zero special case
  if (x === 0) {
    // Return smallest denormalized number with appropriate sign
    return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  }

  // Get the bits of x
  const buffer = new ArrayBuffer(8);
  const float64View = new Float64Array(buffer);
  const int64View = new BigInt64Array(buffer);

  float64View[0] = x;
  let bits = int64View[0]!;

  // Determine if we need to increment or decrement
  const shouldIncrement = (x > 0 && y > x) || (x < 0 && y > x);

  if (shouldIncrement) {
    bits = bits + 1n;
  } else {
    bits = bits - 1n;
  }

  int64View[0] = bits;
  return float64View[0]!;
}

/**
 * Return the distance between x and the nearest adjacent number
 *
 * This is the difference between x and the next representable floating-point value.
 *
 * @param a - Input array storage
 * @returns Array of spacing values
 */
export function spacing(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'spacing', 'spacing is only defined for real numbers.');
  const result = ArrayStorage.zeros(Array.from(a.shape), 'float64');
  const resultData = result.data as Float64Array;
  const thisData = a.data;
  const size = a.size;

  if (isBigIntDType(a.dtype)) {
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      resultData[i] = spacingSingle(Number(typedData[i]));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = spacingSingle(thisData[i] as number);
    }
  }

  return result;
}

/**
 * Compute spacing for a single value
 * @private
 */
function spacingSingle(x: number): number {
  // Handle special cases
  if (Number.isNaN(x)) {
    return NaN;
  }
  if (!Number.isFinite(x)) {
    return NaN;
  }

  const absX = Math.abs(x);

  // For zero, return the smallest positive denormalized number
  if (absX === 0) {
    return Number.MIN_VALUE;
  }

  // The spacing is the absolute difference between x and the next float
  const next = nextafterSingle(x, Infinity);
  return Math.abs(next - x);
}

/**
 * Helper: Create broadcast view of storage to target shape
 * @private
 */
function broadcastToStorage(storage: ArrayStorage, targetShape: readonly number[]): ArrayStorage {
  const ndim = storage.shape.length;
  const targetNdim = targetShape.length;
  const newStrides = new Array(targetNdim).fill(0);

  // Align dimensions from the right
  for (let i = 0; i < ndim; i++) {
    const targetIdx = targetNdim - ndim + i;
    const dim = storage.shape[i]!;
    const targetDim = targetShape[targetIdx]!;

    if (dim === targetDim) {
      newStrides[targetIdx] = storage.strides[i]!;
    } else if (dim === 1) {
      newStrides[targetIdx] = 0; // Broadcast this dimension
    } else {
      throw new Error('Invalid broadcast');
    }
  }

  return ArrayStorage.fromData(
    storage.data,
    Array.from(targetShape),
    storage.dtype,
    newStrides,
    storage.offset
  );
}

// ============================================================
// Additional Logic/Type Checking Operations
// ============================================================

/**
 * Test element-wise for complex number.
 *
 * For complex arrays, returns true for elements with non-zero imaginary part.
 * For real arrays, always returns false.
 *
 * @param a - Input array storage
 * @returns Boolean array
 */
export function iscomplex(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const size = a.size;
  const data = new Uint8Array(size);

  if (isComplexDType(dtype)) {
    // Check if imaginary part is non-zero
    const srcData = a.data as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      data[i] = srcData[i * 2 + 1] !== 0 ? 1 : 0;
    }
  }
  // For real arrays, all elements are false (initialized to 0)

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Check whether object is complex type array.
 *
 * @param a - Input array storage
 * @returns true if dtype is complex64 or complex128
 */
export function iscomplexobj(a: ArrayStorage): boolean {
  return isComplexDType(a.dtype as DType);
}

/**
 * Test element-wise for real number (not complex).
 *
 * For complex arrays, returns true for elements with zero imaginary part.
 * For real arrays, always returns true.
 *
 * @param a - Input array storage
 * @returns Boolean array
 */
export function isreal(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const size = a.size;
  const data = new Uint8Array(size);

  if (isComplexDType(dtype)) {
    // Check if imaginary part is zero
    const srcData = a.data as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      data[i] = srcData[i * 2 + 1] === 0 ? 1 : 0;
    }
  } else {
    // For real arrays, all elements are true
    data.fill(1);
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Check whether object is real type array (not complex).
 *
 * @param a - Input array storage
 * @returns true if dtype is NOT complex64 or complex128
 */
export function isrealobj(a: ArrayStorage): boolean {
  return !isComplexDType(a.dtype as DType);
}

/**
 * Test element-wise for negative infinity
 *
 * For complex numbers: True if either real or imaginary part is -Infinity.
 *
 * @param a - Input array storage
 * @returns Boolean array
 */
export function isneginf(a: ArrayStorage): ArrayStorage {
  throwIfComplex(
    a.dtype,
    'isneginf',
    'This operation is not supported for complex values because it would be ambiguous.'
  );
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isBigIntDType(a.dtype)) {
    // BigInt cannot be -Infinity
    // data is already zeros
  } else {
    for (let i = 0; i < size; i++) {
      const val = thisData[i] as number;
      data[i] = val === -Infinity ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Test element-wise for positive infinity
 *
 * For complex numbers: True if either real or imaginary part is +Infinity.
 *
 * @param a - Input array storage
 * @returns Boolean array
 */
export function isposinf(a: ArrayStorage): ArrayStorage {
  throwIfComplex(
    a.dtype,
    'isposinf',
    'This operation is not supported for complex values because it would be ambiguous.'
  );
  const data = new Uint8Array(a.size);
  const thisData = a.data;
  const size = a.size;

  if (isBigIntDType(a.dtype)) {
    // BigInt cannot be +Infinity
    // data is already zeros
  } else {
    for (let i = 0; i < size; i++) {
      const val = thisData[i] as number;
      data[i] = val === Infinity ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(a.shape), 'bool');
}

/**
 * Check if array is Fortran contiguous (column-major order)
 * @param a - Input array storage
 * @returns true if F-contiguous
 */
export function isfortran(a: ArrayStorage): boolean {
  return a.isFContiguous;
}

/**
 * Returns complex array with complex parts close to zero set to real
 * Since numpy-ts doesn't support complex numbers, returns copy
 * @param a - Input array storage
 * @param _tol - Tolerance (unused, for API compatibility)
 * @returns Copy of input array
 */
export function real_if_close(a: ArrayStorage, tol: number = 100): ArrayStorage {
  const dtype = a.dtype;

  // For complex arrays, check if imaginary part is close to zero
  if (isComplexDType(dtype)) {
    const complexData = a.data as Float64Array | Float32Array;
    const size = a.size;
    const eps = dtype === 'complex64' ? 1.1920929e-7 : 2.220446049250313e-16;
    const threshold = tol * eps;

    // Check if all imaginary parts are close to zero
    let allClose = true;
    for (let i = 0; i < size; i++) {
      const im = complexData[i * 2 + 1]!;
      if (Math.abs(im) > threshold) {
        allClose = false;
        break;
      }
    }

    if (allClose) {
      // Return real part only
      const realDtype = dtype === 'complex64' ? 'float32' : 'float64';
      const result = ArrayStorage.zeros(Array.from(a.shape), realDtype);
      const resultData = result.data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        resultData[i] = complexData[i * 2]!;
      }
      return result;
    }

    // Return complex array as-is
    return a.copy();
  }

  // For non-complex arrays, just return a copy
  return a.copy();
}

/**
 * Check if element is a scalar type
 * @param val - Value to check
 * @returns true if scalar
 */
export function isscalar(val: unknown): boolean {
  return (
    typeof val === 'number' ||
    typeof val === 'bigint' ||
    typeof val === 'boolean' ||
    typeof val === 'string'
  );
}

/**
 * Check if object is iterable
 * @param obj - Object to check
 * @returns true if iterable
 */
export function iterable(obj: unknown): boolean {
  if (obj === null || obj === undefined) {
    return false;
  }
  return typeof (obj as Record<symbol, unknown>)[Symbol.iterator] === 'function';
}

/**
 * Check if dtype meets specified criteria
 * @param dtype - Dtype to check
 * @param kind - Kind of dtype ('b' bool, 'i' int, 'u' uint, 'f' float)
 * @returns true if dtype matches kind
 */
export function isdtype(dtype: DType, kind: string): boolean {
  const kindMap: Record<string, DType[]> = {
    b: ['bool'],
    i: ['int8', 'int16', 'int32', 'int64'],
    u: ['uint8', 'uint16', 'uint32', 'uint64'],
    f: ['float32', 'float64'],
  };

  const matchingTypes = kindMap[kind];
  if (!matchingTypes) {
    return false;
  }

  return matchingTypes.includes(dtype);
}

/**
 * Find the dtype that can represent both input dtypes
 * @param dtype1 - First dtype
 * @param dtype2 - Second dtype
 * @returns Promoted dtype
 */
export function promote_types(dtype1: DType, dtype2: DType): DType {
  // NumPy promotion hierarchy (from highest to lowest precedence)
  const hierarchy: DType[] = [
    'float64',
    'float32',
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

  const idx1 = hierarchy.indexOf(dtype1);
  const idx2 = hierarchy.indexOf(dtype2);

  // Return the one with higher precedence (lower index)
  return idx1 <= idx2 ? dtype1 : dtype2;
}
