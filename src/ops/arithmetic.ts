/**
 * Arithmetic operations
 *
 * Pure functions for element-wise arithmetic operations:
 * add, subtract, multiply, divide
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType, isComplexDType, getComplexComponentDType, promoteDTypes } from '../core/dtype';
import { elementwiseBinaryOp } from '../internal/compute';

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

/**
 * Add two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function add(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return addScalar(a, b);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return addArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x + y, 'add');
}

/**
 * Fast path for adding two contiguous arrays
 * @private
 */
function addArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal = typeof aData[i] === 'bigint' ? aData[i] : BigInt(Math.round(Number(aData[i])));
        const bVal = typeof bData[i] === 'bigint' ? bData[i] : BigInt(Math.round(Number(bData[i])));
        resultTyped[i] = (aVal as bigint) + (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[i]! + bTyped[i]!;
      }
    }
  } else {
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal = typeof aData[i] === 'bigint' ? Number(aData[i]) : (aData[i] as number);
        const bVal = typeof bData[i] === 'bigint' ? Number(bData[i]) : (bData[i] as number);
        resultData[i] = aVal + bVal;
      }
    } else {
      // Pure numeric operations - fully optimizable
      for (let i = 0; i < size; i++) {
        resultData[i] = (aData[i] as number) + (bData[i] as number);
      }
    }
  }

  return result;
}

/**
 * Subtract two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function subtract(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return subtractScalar(a, b);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return subtractArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x - y, 'subtract');
}

/**
 * Fast path for subtracting two contiguous arrays
 * @private
 */
function subtractArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal = typeof aData[i] === 'bigint' ? aData[i] : BigInt(Math.round(Number(aData[i])));
        const bVal = typeof bData[i] === 'bigint' ? bData[i] : BigInt(Math.round(Number(bData[i])));
        resultTyped[i] = (aVal as bigint) - (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[i]! - bTyped[i]!;
      }
    }
  } else {
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal = typeof aData[i] === 'bigint' ? Number(aData[i]) : (aData[i] as number);
        const bVal = typeof bData[i] === 'bigint' ? Number(bData[i]) : (bData[i] as number);
        resultData[i] = aVal - bVal;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = (aData[i] as number) - (bData[i] as number);
      }
    }
  }

  return result;
}

/**
 * Multiply two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function multiply(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return multiplyScalar(a, b);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return multiplyArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x * y, 'multiply');
}

/**
 * Fast path for multiplying two contiguous arrays
 * @private
 */
function multiplyArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal = typeof aData[i] === 'bigint' ? aData[i] : BigInt(Math.round(Number(aData[i])));
        const bVal = typeof bData[i] === 'bigint' ? bData[i] : BigInt(Math.round(Number(bData[i])));
        resultTyped[i] = (aVal as bigint) * (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[i]! * bTyped[i]!;
      }
    }
  } else {
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal = typeof aData[i] === 'bigint' ? Number(aData[i]) : (aData[i] as number);
        const bVal = typeof bData[i] === 'bigint' ? Number(bData[i]) : (bData[i] as number);
        resultData[i] = aVal * bVal;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = (aData[i] as number) * (bData[i] as number);
      }
    }
  }

  return result;
}

/**
 * Divide two arrays or array and scalar
 *
 * NumPy behavior: Integer division always promotes to float
 * Type promotion rules:
 * - float64 + anything → float64
 * - float32 + integer → float32
 * - integer + integer → float64
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage with promoted float dtype
 */
export function divide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return divideScalar(a, b);
  }

  // Determine result dtype using NumPy promotion rules
  const aIsFloat64 = a.dtype === 'float64';
  const bIsFloat64 = b.dtype === 'float64';
  const aIsFloat32 = a.dtype === 'float32';
  const bIsFloat32 = b.dtype === 'float32';

  // If either is float64, result is float64
  if (aIsFloat64 || bIsFloat64) {
    const aFloat = aIsFloat64 ? a : convertToFloatDType(a, 'float64');
    const bFloat = bIsFloat64 ? b : convertToFloatDType(b, 'float64');
    return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
  }

  // If either is float32, result is float32
  if (aIsFloat32 || bIsFloat32) {
    const aFloat = aIsFloat32 ? a : convertToFloatDType(a, 'float32');
    const bFloat = bIsFloat32 ? b : convertToFloatDType(b, 'float32');
    return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
  }

  // Both are integers, promote to float64
  const aFloat = convertToFloatDType(a, 'float64');
  const bFloat = convertToFloatDType(b, 'float64');
  return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
}

/**
 * Convert ArrayStorage to float dtype
 * @private
 */
function convertToFloatDType(
  storage: ArrayStorage,
  targetDtype: 'float32' | 'float64'
): ArrayStorage {
  const result = ArrayStorage.zeros(Array.from(storage.shape), targetDtype);
  const size = storage.size;
  const srcData = storage.data;
  const dstData = result.data;

  for (let i = 0; i < size; i++) {
    dstData[i] = Number(srcData[i]!);
  }

  return result;
}

/**
 * Add scalar to array (optimized path)
 * @private
 */
function addScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! + scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) + scalar;
    }
  }

  return result;
}

/**
 * Subtract scalar from array (optimized path)
 * @private
 */
function subtractScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! - scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) - scalar;
    }
  }

  return result;
}

/**
 * Multiply array by scalar (optimized path)
 * @private
 */
function multiplyScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! * scalarBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) * scalar;
    }
  }

  return result;
}

/**
 * Divide array by scalar (optimized path)
 * NumPy behavior: Integer division promotes to float64
 * @private
 */
function divideScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // NumPy behavior: Integer division always promotes to float64
  // This allows representing inf/nan for division by zero
  // Bool is also promoted to float64 (NumPy behavior)
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const resultDtype = isIntegerType ? 'float64' : dtype;

  // Create result with promoted dtype
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // Convert BigInt to Number for division (promotes to float64)
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) / scalar;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(data[i]!) / scalar;
    }
  }

  return result;
}

/**
 * Absolute value of each element
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with absolute values
 */
export function absolute(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // For complex types, result is the component dtype (magnitude is real)
  if (isComplexDType(dtype)) {
    const resultDtype = getComplexComponentDType(dtype);
    const result = ArrayStorage.zeros(shape, resultDtype);
    const resultData = result.data as Float64Array | Float32Array;
    const srcData = data as Float64Array | Float32Array;

    // Data is interleaved [re, im, re, im, ...]
    // |z| = sqrt(re² + im²)
    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;
      resultData[i] = Math.sqrt(re * re + im * im);
    }

    return result;
  }

  // Create result with same dtype for non-complex
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = thisTyped[i]!;
      resultTyped[i] = val < 0n ? -val : val;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.abs(Number(data[i]!));
    }
  }

  return result;
}

/**
 * Numerical negative (element-wise negation)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with negated values
 */
export function negative(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      resultTyped[i] = -thisTyped[i]!;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = -Number(data[i]!);
    }
  }

  return result;
}

/**
 * Sign of each element (-1, 0, or 1)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with signs
 */
export function sign(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = thisTyped[i]!;
      resultTyped[i] = val > 0n ? 1n : val < 0n ? -1n : 0n;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]!);
      resultData[i] = val > 0 ? 1 : val < 0 ? -1 : 0;
    }
  }

  return result;
}

/**
 * Modulo operation (remainder after division)
 * NumPy behavior: Uses floor modulo (sign follows divisor), not JavaScript's truncate modulo
 * Preserves dtype for integer types
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with modulo values
 */
export function mod(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return modScalar(a, b);
  }
  // NumPy uses floor modulo: ((x % y) + y) % y for proper sign handling
  return elementwiseBinaryOp(a, b, (x, y) => ((x % y) + y) % y, 'mod');
}

/**
 * Modulo with scalar divisor (optimized path)
 * NumPy uses floor modulo: result has same sign as divisor
 * @private
 */
function modScalar(storage: ArrayStorage, divisor: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Create result with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt arithmetic - use floor modulo
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const divisorBig = BigInt(Math.round(divisor));
    for (let i = 0; i < size; i++) {
      const val = thisTyped[i]!;
      // Floor modulo for BigInt
      resultTyped[i] = ((val % divisorBig) + divisorBig) % divisorBig;
    }
  } else {
    // Regular numeric types - use floor modulo
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]!);
      // Floor modulo: ((x % y) + y) % y
      resultData[i] = ((val % divisor) + divisor) % divisor;
    }
  }

  return result;
}

/**
 * Floor division (division with result rounded down)
 * NumPy behavior: Preserves integer types
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with floor division values
 */
export function floorDivide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return floorDivideScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => Math.floor(x / y), 'floor_divide');
}

/**
 * Floor division with scalar divisor (optimized path)
 * @private
 */
function floorDivideScalar(storage: ArrayStorage, divisor: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // NumPy behavior: floor_divide preserves integer types
  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt floor division
    const thisTyped = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const divisorBig = BigInt(Math.round(divisor));
    for (let i = 0; i < size; i++) {
      resultTyped[i] = thisTyped[i]! / divisorBig;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.floor(Number(data[i]!) / divisor);
    }
  }

  return result;
}

/**
 * Unary positive (returns a copy of the array)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage (copy of input)
 */
export function positive(a: ArrayStorage): ArrayStorage {
  // Positive is essentially a no-op that returns a copy
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  // Copy data
  for (let i = 0; i < size; i++) {
    resultData[i] = data[i]!;
  }

  return result;
}

/**
 * Reciprocal (1/x) of each element
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with reciprocal values
 */
export function reciprocal(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // NumPy behavior: reciprocal always promotes integers to float64
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const resultDtype = isIntegerType ? 'float64' : dtype;

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    // BigInt input promotes to float64
    for (let i = 0; i < size; i++) {
      resultData[i] = 1.0 / Number(data[i]!);
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = 1.0 / Number(data[i]!);
    }
  }

  return result;
}

/**
 * Cube root of each element
 * NumPy behavior: Promotes integer types to float64
 *
 * @param a - Input array storage
 * @returns Result storage with cube root values
 */
export function cbrt(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // NumPy behavior: cbrt always promotes integers to float64
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const resultDtype = isIntegerType ? 'float64' : dtype;

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  // Handle all types uniformly - convert to number and apply Math.cbrt
  for (let i = 0; i < size; i++) {
    resultData[i] = Math.cbrt(Number(data[i]!));
  }

  return result;
}

/**
 * Absolute value of each element, returning float
 * NumPy behavior: fabs always returns floating point
 *
 * @param a - Input array storage
 * @returns Result storage with absolute values as float
 */
export function fabs(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // fabs always returns float64, except for float32 which stays float32
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  // Handle all types uniformly
  for (let i = 0; i < size; i++) {
    resultData[i] = Math.abs(Number(data[i]!));
  }

  return result;
}

/**
 * Returns both quotient and remainder (floor divide and modulo)
 * NumPy behavior: divmod(a, b) = (floor_divide(a, b), mod(a, b))
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Tuple of [quotient, remainder] storages
 */
export function divmod(a: ArrayStorage, b: ArrayStorage | number): [ArrayStorage, ArrayStorage] {
  const quotient = floorDivide(a, b);
  const remainder = mod(a, b);
  return [quotient, remainder];
}

/**
 * Element-wise square of each element
 * NumPy behavior: x**2
 *
 * @param a - Input array storage
 * @returns Result storage with squared values
 */
export function square(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const bigData = data as BigInt64Array | BigUint64Array;
    const bigResultData = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      bigResultData[i] = bigData[i]! * bigData[i]!;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]!);
      resultData[i] = val * val;
    }
  }

  return result;
}

/**
 * Remainder of division (same as mod)
 * NumPy behavior: Same as mod, alias for compatibility
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with remainder values
 */
export function remainder(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  return mod(a, b);
}

/**
 * Heaviside step function
 * NumPy behavior:
 *   heaviside(x1, x2) = 0 if x1 < 0
 *                     = x2 if x1 == 0
 *                     = 1 if x1 > 0
 *
 * @param x1 - Input array storage
 * @param x2 - Value to use when x1 == 0 (array storage or scalar)
 * @returns Result storage with heaviside values
 */
export function heaviside(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  const dtype = x1.dtype;
  const shape = Array.from(x1.shape);
  const size = x1.size;

  // Result is always float64
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (typeof x2 === 'number') {
    // Scalar x2
    for (let i = 0; i < size; i++) {
      const val = Number(x1.data[i]!);
      if (val < 0) {
        resultData[i] = 0;
      } else if (val === 0) {
        resultData[i] = x2;
      } else {
        resultData[i] = 1;
      }
    }
  } else {
    // Array x2 - needs to broadcast
    const x2Data = x2.data;
    const x2Shape = x2.shape;

    // Simple case: same shape
    if (shape.every((d, i) => d === x2Shape[i])) {
      for (let i = 0; i < size; i++) {
        const val = Number(x1.data[i]!);
        if (val < 0) {
          resultData[i] = 0;
        } else if (val === 0) {
          resultData[i] = Number(x2Data[i]!);
        } else {
          resultData[i] = 1;
        }
      }
    } else {
      // Broadcasting case - use elementwiseBinaryOp approach
      for (let i = 0; i < size; i++) {
        const val = Number(x1.data[i]!);
        // Simple broadcast: assume x2 is broadcastable to x1
        const x2Idx = i % x2.size;
        if (val < 0) {
          resultData[i] = 0;
        } else if (val === 0) {
          resultData[i] = Number(x2Data[x2Idx]!);
        } else {
          resultData[i] = 1;
        }
      }
    }
  }

  return result;
}

/**
 * First array raised to power of second, always promoting to float
 * @param x1 - Base values
 * @param x2 - Exponent values
 * @returns Result in float64
 */
export function float_power(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  if (typeof x2 === 'number') {
    const result = ArrayStorage.zeros(Array.from(x1.shape), 'float64');
    const resultData = result.data as Float64Array;
    const x1Data = x1.data;
    const size = x1.size;

    for (let i = 0; i < size; i++) {
      resultData[i] = Math.pow(Number(x1Data[i]!), x2);
    }

    return result;
  }

  return elementwiseBinaryOp(x1, x2, (a, b) => Math.pow(a, b), 'float_power');
}

/**
 * Element-wise remainder of division (fmod)
 * Unlike mod/remainder, fmod matches C fmod behavior
 * @param x1 - Dividend
 * @param x2 - Divisor
 * @returns Remainder
 */
export function fmod(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  if (typeof x2 === 'number') {
    const result = x1.copy();
    const resultData = result.data;
    const size = x1.size;

    for (let i = 0; i < size; i++) {
      const val = Number(resultData[i]!);
      resultData[i] = val - Math.trunc(val / x2) * x2;
    }

    return result;
  }

  return elementwiseBinaryOp(x1, x2, (a, b) => a - Math.trunc(a / b) * b, 'fmod');
}

/**
 * Decompose floating point numbers into mantissa and exponent
 * Returns [mantissa, exponent] where x = mantissa * 2^exponent
 * @param x - Input array
 * @returns Tuple of [mantissa, exponent] arrays
 */
export function frexp(x: ArrayStorage): [ArrayStorage, ArrayStorage] {
  const mantissa = ArrayStorage.zeros(Array.from(x.shape), 'float64');
  const exponent = ArrayStorage.zeros(Array.from(x.shape), 'int32');
  const mantissaData = mantissa.data as Float64Array;
  const exponentData = exponent.data as Int32Array;
  const xData = x.data;
  const size = x.size;

  for (let i = 0; i < size; i++) {
    const val = Number(xData[i]!);

    if (val === 0 || !isFinite(val)) {
      mantissaData[i] = val;
      exponentData[i] = 0;
    } else {
      const exp = Math.floor(Math.log2(Math.abs(val))) + 1;
      const man = val / Math.pow(2, exp);
      mantissaData[i] = man;
      exponentData[i] = exp;
    }
  }

  return [mantissa, exponent];
}

/**
 * Greatest common divisor
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns GCD
 */
export function gcd(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  const gcdSingle = (a: number, b: number): number => {
    a = Math.abs(Math.trunc(a));
    b = Math.abs(Math.trunc(b));
    while (b !== 0) {
      const temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  };

  if (typeof x2 === 'number') {
    const result = ArrayStorage.zeros(Array.from(x1.shape), 'int32');
    const resultData = result.data as Int32Array;
    const x1Data = x1.data;
    const size = x1.size;
    const x2Int = Math.abs(Math.trunc(x2));

    for (let i = 0; i < size; i++) {
      resultData[i] = gcdSingle(Number(x1Data[i]!), x2Int);
    }

    return result;
  }

  // Array case - use elementwiseBinaryOp then convert to int32
  const tempResult = elementwiseBinaryOp(x1, x2, gcdSingle, 'gcd');

  // Convert result to int32
  const result = ArrayStorage.zeros(Array.from(tempResult.shape), 'int32');
  const resultData = result.data as Int32Array;
  const tempData = tempResult.data;
  const size = tempResult.size;

  for (let i = 0; i < size; i++) {
    resultData[i] = Math.round(Number(tempData[i]!));
  }

  return result;
}

/**
 * Least common multiple
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns LCM
 */
export function lcm(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  const gcdSingle = (a: number, b: number): number => {
    a = Math.abs(Math.trunc(a));
    b = Math.abs(Math.trunc(b));
    while (b !== 0) {
      const temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  };

  const lcmSingle = (a: number, b: number): number => {
    a = Math.abs(Math.trunc(a));
    b = Math.abs(Math.trunc(b));
    if (a === 0 || b === 0) return 0;
    return (a * b) / gcdSingle(a, b);
  };

  if (typeof x2 === 'number') {
    const result = ArrayStorage.zeros(Array.from(x1.shape), 'int32');
    const resultData = result.data as Int32Array;
    const x1Data = x1.data;
    const size = x1.size;
    const x2Int = Math.abs(Math.trunc(x2));

    for (let i = 0; i < size; i++) {
      resultData[i] = lcmSingle(Number(x1Data[i]!), x2Int);
    }

    return result;
  }

  // Array case - use elementwiseBinaryOp then convert to int32
  const tempResult = elementwiseBinaryOp(x1, x2, lcmSingle, 'lcm');

  // Convert result to int32
  const result = ArrayStorage.zeros(Array.from(tempResult.shape), 'int32');
  const resultData = result.data as Int32Array;
  const tempData = tempResult.data;
  const size = tempResult.size;

  for (let i = 0; i < size; i++) {
    resultData[i] = Math.round(Number(tempData[i]!));
  }

  return result;
}

/**
 * Returns x1 * 2^x2, element-wise
 * @param x1 - Mantissa
 * @param x2 - Exponent
 * @returns Result
 */
export function ldexp(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  if (typeof x2 === 'number') {
    const result = ArrayStorage.zeros(Array.from(x1.shape), 'float64');
    const resultData = result.data as Float64Array;
    const x1Data = x1.data;
    const size = x1.size;
    const multiplier = Math.pow(2, x2);

    for (let i = 0; i < size; i++) {
      resultData[i] = Number(x1Data[i]!) * multiplier;
    }

    return result;
  }

  return elementwiseBinaryOp(x1, x2, (a, b) => a * Math.pow(2, b), 'ldexp');
}

/**
 * Return fractional and integral parts of array
 * @param x - Input array
 * @returns Tuple of [fractional, integral] arrays
 */
export function modf(x: ArrayStorage): [ArrayStorage, ArrayStorage] {
  const fractional = ArrayStorage.zeros(Array.from(x.shape), 'float64');
  const integral = ArrayStorage.zeros(Array.from(x.shape), 'float64');
  const fractionalData = fractional.data as Float64Array;
  const integralData = integral.data as Float64Array;
  const xData = x.data;
  const size = x.size;

  for (let i = 0; i < size; i++) {
    const val = Number(xData[i]!);
    const intPart = Math.trunc(val);
    integralData[i] = intPart;
    fractionalData[i] = val - intPart;
  }

  return [fractional, integral];
}
