/**
 * Bitwise operations
 *
 * Pure functions for element-wise bitwise operations:
 * bitwise_and, bitwise_or, bitwise_xor, bitwise_not, invert,
 * left_shift, right_shift, packbits, unpackbits
 *
 * These operations only work on integer types.
 */

import { ArrayStorage } from '../storage';
import { isBigIntDType, isIntegerDType, promoteDTypes, DType } from '../dtype';
import { elementwiseBinaryOp } from '../internal/compute';

/**
 * Helper: Validate that dtype is an integer type for bitwise operations
 */
function validateIntegerDType(dtype: DType, opName: string): void {
  if (!isIntegerDType(dtype) && dtype !== 'bool') {
    throw new TypeError(
      `ufunc '${opName}' not supported for the input types, and the inputs could not be safely coerced to any supported types`
    );
  }
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

/**
 * Bitwise AND of two arrays or array and scalar
 *
 * @param a - First array storage (must be integer type)
 * @param b - Second array storage or scalar (must be integer type)
 * @returns Result storage
 */
export function bitwise_and(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  validateIntegerDType(a.dtype, 'bitwise_and');

  if (typeof b === 'number') {
    return bitwiseAndScalar(a, b);
  }

  validateIntegerDType(b.dtype, 'bitwise_and');

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return bitwiseAndArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x & y, 'bitwise_and');
}

/**
 * Fast path for bitwise AND of two contiguous arrays
 * @private
 */
function bitwiseAndArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? aData[aOff + i]
            : BigInt(Math.round(Number(aData[aOff + i])));
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? bData[bOff + i]
            : BigInt(Math.round(Number(bData[bOff + i])));
        resultTyped[i] = (aVal as bigint) & (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[aOff + i]! & bTyped[bOff + i]!;
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = (aData[aOff + i] as number) & (bData[bOff + i] as number);
    }
  }

  return result;
}

/**
 * Bitwise AND with scalar (optimized path)
 * @private
 */
function bitwiseAndScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const off = storage.offset;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[i]! & scalarBig;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[off + i]! & scalarBig;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) & scalarBig;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[i] as number) & scalar;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[off + i] as number) & scalar;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) & scalar;
      }
    }
  }

  return result;
}

/**
 * Bitwise OR of two arrays or array and scalar
 *
 * @param a - First array storage (must be integer type)
 * @param b - Second array storage or scalar (must be integer type)
 * @returns Result storage
 */
export function bitwise_or(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  validateIntegerDType(a.dtype, 'bitwise_or');

  if (typeof b === 'number') {
    return bitwiseOrScalar(a, b);
  }

  validateIntegerDType(b.dtype, 'bitwise_or');

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return bitwiseOrArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x | y, 'bitwise_or');
}

/**
 * Fast path for bitwise OR of two contiguous arrays
 * @private
 */
function bitwiseOrArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? aData[aOff + i]
            : BigInt(Math.round(Number(aData[aOff + i])));
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? bData[bOff + i]
            : BigInt(Math.round(Number(bData[bOff + i])));
        resultTyped[i] = (aVal as bigint) | (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[aOff + i]! | bTyped[bOff + i]!;
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = (aData[aOff + i] as number) | (bData[bOff + i] as number);
    }
  }

  return result;
}

/**
 * Bitwise OR with scalar (optimized path)
 * @private
 */
function bitwiseOrScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const off = storage.offset;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[i]! | scalarBig;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[off + i]! | scalarBig;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) | scalarBig;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[i] as number) | scalar;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[off + i] as number) | scalar;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) | scalar;
      }
    }
  }

  return result;
}

/**
 * Bitwise XOR of two arrays or array and scalar
 *
 * @param a - First array storage (must be integer type)
 * @param b - Second array storage or scalar (must be integer type)
 * @returns Result storage
 */
export function bitwise_xor(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  validateIntegerDType(a.dtype, 'bitwise_xor');

  if (typeof b === 'number') {
    return bitwiseXorScalar(a, b);
  }

  validateIntegerDType(b.dtype, 'bitwise_xor');

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return bitwiseXorArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x ^ y, 'bitwise_xor');
}

/**
 * Fast path for bitwise XOR of two contiguous arrays
 * @private
 */
function bitwiseXorArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? aData[aOff + i]
            : BigInt(Math.round(Number(aData[aOff + i])));
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? bData[bOff + i]
            : BigInt(Math.round(Number(bData[bOff + i])));
        resultTyped[i] = (aVal as bigint) ^ (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[aOff + i]! ^ bTyped[bOff + i]!;
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = (aData[aOff + i] as number) ^ (bData[bOff + i] as number);
    }
  }

  return result;
}

/**
 * Bitwise XOR with scalar (optimized path)
 * @private
 */
function bitwiseXorScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const off = storage.offset;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[i]! ^ scalarBig;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[off + i]! ^ scalarBig;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) ^ scalarBig;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[i] as number) ^ scalar;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[off + i] as number) ^ scalar;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) ^ scalar;
      }
    }
  }

  return result;
}

/**
 * Bitwise NOT (invert) of each element
 *
 * @param a - Input array storage (must be integer type)
 * @returns Result storage with bitwise NOT values
 */
export function bitwise_not(a: ArrayStorage): ArrayStorage {
  validateIntegerDType(a.dtype, 'bitwise_not');

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const off = a.offset;
  const size = a.size;
  const contiguous = a.isCContiguous;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = ~thisTyped[i]!;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = ~thisTyped[off + i]!;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = ~(a.iget(i) as bigint);
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = ~(data[i] as number);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = ~(data[off + i] as number);
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = ~Number(a.iget(i));
      }
    }
  }

  return result;
}

/**
 * Invert (bitwise NOT) - alias for bitwise_not
 *
 * @param a - Input array storage (must be integer type)
 * @returns Result storage with inverted values
 */
export function invert(a: ArrayStorage): ArrayStorage {
  return bitwise_not(a);
}

/**
 * Left shift of array elements
 *
 * @param a - Input array storage (must be integer type)
 * @param b - Shift amount (array storage or scalar)
 * @returns Result storage with left-shifted values
 */
export function left_shift(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  validateIntegerDType(a.dtype, 'left_shift');

  if (typeof b === 'number') {
    return leftShiftScalar(a, b);
  }

  validateIntegerDType(b.dtype, 'left_shift');

  // Fast path: single-element array or broadcastable scalar shape treated as scalar
  if (b.size === 1 || (b.ndim === 1 && b.shape[0] === 1)) {
    const shiftVal = Number(b.iget(0));
    return leftShiftScalar(a, shiftVal);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return leftShiftArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x << y, 'left_shift');
}

/**
 * Fast path for left shift of two contiguous arrays
 * @private
 */
function leftShiftArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const aVal =
        typeof aData[aOff + i] === 'bigint'
          ? aData[aOff + i]
          : BigInt(Math.round(Number(aData[aOff + i])));
      const bVal =
        typeof bData[bOff + i] === 'bigint'
          ? bData[bOff + i]
          : BigInt(Math.round(Number(bData[bOff + i])));
      resultTyped[i] = (aVal as bigint) << (bVal as bigint);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = (aData[aOff + i] as number) << (bData[bOff + i] as number);
    }
  }

  return result;
}

/**
 * Left shift with scalar (optimized path)
 * @private
 */
function leftShiftScalar(storage: ArrayStorage, shift: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const off = storage.offset;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const shiftBig = BigInt(Math.round(shift));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[i]! << shiftBig;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[off + i]! << shiftBig;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) << shiftBig;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[i] as number) << shift;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[off + i] as number) << shift;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) << shift;
      }
    }
  }

  return result;
}

/**
 * Right shift of array elements
 *
 * @param a - Input array storage (must be integer type)
 * @param b - Shift amount (array storage or scalar)
 * @returns Result storage with right-shifted values
 */
export function right_shift(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  validateIntegerDType(a.dtype, 'right_shift');

  if (typeof b === 'number') {
    return rightShiftScalar(a, b);
  }

  validateIntegerDType(b.dtype, 'right_shift');

  // Fast path: single-element array or broadcastable scalar shape treated as scalar
  if (b.size === 1 || (b.ndim === 1 && b.shape[0] === 1)) {
    const shiftVal = Number(b.iget(0));
    return rightShiftScalar(a, shiftVal);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    return rightShiftArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x >> y, 'right_shift');
}

/**
 * Fast path for right shift of two contiguous arrays
 * @private
 */
function rightShiftArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const aVal =
        typeof aData[aOff + i] === 'bigint'
          ? aData[aOff + i]
          : BigInt(Math.round(Number(aData[aOff + i])));
      const bVal =
        typeof bData[bOff + i] === 'bigint'
          ? bData[bOff + i]
          : BigInt(Math.round(Number(bData[bOff + i])));
      resultTyped[i] = (aVal as bigint) >> (bVal as bigint);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = (aData[aOff + i] as number) >> (bData[bOff + i] as number);
    }
  }

  return result;
}

/**
 * Right shift with scalar (optimized path)
 * @private
 */
function rightShiftScalar(storage: ArrayStorage, shift: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const off = storage.offset;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.zeros(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const shiftBig = BigInt(Math.round(shift));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[i]! >> shiftBig;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = thisTyped[off + i]! >> shiftBig;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) >> shiftBig;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[i] as number) >> shift;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (data[off + i] as number) >> shift;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) >> shift;
      }
    }
  }

  return result;
}

/**
 * Pack binary values into uint8 array
 *
 * Packs the elements of a binary-valued array into bits in a uint8 array.
 * The result has the same shape as the input, except for the specified axis
 * which is divided by 8 (rounded up).
 *
 * @param a - Input array (values are interpreted as binary: 0 or non-zero)
 * @param axis - The dimension over which bit-packing is done (default: -1, meaning the last axis)
 * @param bitorder - The order of bits within each packed byte. 'big' means the most significant bit is first. (default: 'big')
 * @returns Packed uint8 array
 */
export function packbits(
  a: ArrayStorage,
  axis: number = -1,
  bitorder: 'big' | 'little' = 'big'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Handle negative axis
  if (axis < 0) {
    axis = ndim + axis;
  }

  if (axis < 0 || axis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Calculate output shape
  const axisSize = shape[axis]!;
  const packedAxisSize = Math.ceil(axisSize / 8);
  const outShape = [...shape];
  outShape[axis] = packedAxisSize;

  const result = ArrayStorage.zeros(outShape, 'uint8');
  const resultData = result.data as Uint8Array;

  const aOff = a.offset;
  const aContiguous = a.isCContiguous;

  // For 1D arrays, simple case
  if (ndim === 1) {
    for (let i = 0; i < packedAxisSize; i++) {
      let byte = 0;
      for (let bit = 0; bit < 8; bit++) {
        const srcIdx = i * 8 + bit;
        if (srcIdx < axisSize) {
          const val = aContiguous
            ? Number(a.data[aOff + srcIdx]!) !== 0
              ? 1
              : 0
            : Number(a.iget(srcIdx)) !== 0
              ? 1
              : 0;
          if (bitorder === 'big') {
            byte |= val << (7 - bit);
          } else {
            byte |= val << bit;
          }
        }
      }
      resultData[i] = byte;
    }
    return result;
  }

  // For N-D arrays, iterate over all combinations except the packed axis
  const preAxisShape = shape.slice(0, axis);
  const postAxisShape = shape.slice(axis + 1);

  const preAxisSize = preAxisShape.reduce((acc, dim) => acc * dim, 1);
  const postAxisSize = postAxisShape.reduce((acc, dim) => acc * dim, 1);

  // Calculate strides for input and output
  const inputStrides = computeStrides(shape);
  const outputStrides = computeStrides(outShape);

  for (let pre = 0; pre < preAxisSize; pre++) {
    for (let post = 0; post < postAxisSize; post++) {
      for (let packedIdx = 0; packedIdx < packedAxisSize; packedIdx++) {
        let byte = 0;
        for (let bit = 0; bit < 8; bit++) {
          const axisIdx = packedIdx * 8 + bit;
          if (axisIdx < axisSize) {
            // Calculate input linear index
            let inputIdx = 0;
            let preRemaining = pre;
            for (let d = 0; d < axis; d++) {
              const dimSize =
                d < axis - 1 ? preAxisShape.slice(d + 1).reduce((acc, b) => acc * b, 1) : 1;
              const coord = Math.floor(preRemaining / dimSize);
              preRemaining %= dimSize;
              inputIdx += coord * inputStrides[d]!;
            }
            inputIdx += axisIdx * inputStrides[axis]!;
            let postRemaining = post;
            for (let d = axis + 1; d < ndim; d++) {
              const dimSize =
                d < ndim - 1 ? postAxisShape.slice(d - axis).reduce((acc, b) => acc * b, 1) : 1;
              const coord = Math.floor(postRemaining / dimSize);
              postRemaining %= dimSize;
              inputIdx += coord * inputStrides[d]!;
            }

            const val = aContiguous
              ? Number(a.data[aOff + inputIdx]!) !== 0
                ? 1
                : 0
              : Number(a.iget(inputIdx)) !== 0
                ? 1
                : 0;
            if (bitorder === 'big') {
              byte |= val << (7 - bit);
            } else {
              byte |= val << bit;
            }
          }
        }

        // Calculate output linear index
        let outputIdx = 0;
        let preRemaining = pre;
        for (let d = 0; d < axis; d++) {
          const dimSize =
            d < axis - 1 ? preAxisShape.slice(d + 1).reduce((acc, b) => acc * b, 1) : 1;
          const coord = Math.floor(preRemaining / dimSize);
          preRemaining %= dimSize;
          outputIdx += coord * outputStrides[d]!;
        }
        outputIdx += packedIdx * outputStrides[axis]!;
        let postRemaining = post;
        for (let d = axis + 1; d < ndim; d++) {
          const dimSize =
            d < ndim - 1 ? postAxisShape.slice(d - axis).reduce((acc, b) => acc * b, 1) : 1;
          const coord = Math.floor(postRemaining / dimSize);
          postRemaining %= dimSize;
          outputIdx += coord * outputStrides[d]!;
        }

        resultData[outputIdx] = byte;
      }
    }
  }

  return result;
}

/**
 * Unpack uint8 array into binary values
 *
 * Unpacks elements of a uint8 array into a binary-valued output array.
 * Each element of the input array is unpacked into 8 binary values.
 *
 * @param a - Input uint8 array
 * @param axis - The dimension over which bit-unpacking is done (default: -1, meaning the last axis)
 * @param count - The number of elements to unpack along axis, or -1 for all (default: -1)
 * @param bitorder - The order of bits within each packed byte. 'big' means the most significant bit is first. (default: 'big')
 * @returns Unpacked uint8 array of 0s and 1s
 */
export function unpackbits(
  a: ArrayStorage,
  axis: number = -1,
  count: number = -1,
  bitorder: 'big' | 'little' = 'big'
): ArrayStorage {
  if (a.dtype !== 'uint8') {
    throw new TypeError('Expected an input array of unsigned byte data type');
  }

  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Handle negative axis
  if (axis < 0) {
    axis = ndim + axis;
  }

  if (axis < 0 || axis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Calculate output shape
  const packedAxisSize = shape[axis]!;
  let unpackedAxisSize = packedAxisSize * 8;

  // Apply count if specified
  if (count >= 0) {
    unpackedAxisSize = count;
  }

  const outShape = [...shape];
  outShape[axis] = unpackedAxisSize;

  const result = ArrayStorage.zeros(outShape, 'uint8');
  const resultData = result.data as Uint8Array;

  const aOff = a.offset;
  const aContiguous = a.isCContiguous;

  // For 1D arrays, simple case
  if (ndim === 1) {
    for (let i = 0; i < packedAxisSize; i++) {
      const byte = aContiguous ? Number(a.data[aOff + i]!) : Number(a.iget(i));
      for (let bit = 0; bit < 8; bit++) {
        const outIdx = i * 8 + bit;
        if (outIdx >= unpackedAxisSize) break;
        if (bitorder === 'big') {
          resultData[outIdx] = (byte >> (7 - bit)) & 1;
        } else {
          resultData[outIdx] = (byte >> bit) & 1;
        }
      }
    }
    return result;
  }

  // For N-D arrays
  const preAxisShape = shape.slice(0, axis);
  const postAxisShape = shape.slice(axis + 1);

  const preAxisSize = preAxisShape.reduce((acc, dim) => acc * dim, 1);
  const postAxisSize = postAxisShape.reduce((acc, dim) => acc * dim, 1);

  const inputStrides = computeStrides(shape);
  const outputStrides = computeStrides(outShape);

  for (let pre = 0; pre < preAxisSize; pre++) {
    for (let post = 0; post < postAxisSize; post++) {
      for (let packedIdx = 0; packedIdx < packedAxisSize; packedIdx++) {
        // Calculate input linear index
        let inputIdx = 0;
        let preRemaining = pre;
        for (let d = 0; d < axis; d++) {
          const dimSize =
            d < axis - 1 ? preAxisShape.slice(d + 1).reduce((acc, b) => acc * b, 1) : 1;
          const coord = Math.floor(preRemaining / dimSize);
          preRemaining %= dimSize;
          inputIdx += coord * inputStrides[d]!;
        }
        inputIdx += packedIdx * inputStrides[axis]!;
        let postRemaining = post;
        for (let d = axis + 1; d < ndim; d++) {
          const dimSize =
            d < ndim - 1 ? postAxisShape.slice(d - axis).reduce((acc, b) => acc * b, 1) : 1;
          const coord = Math.floor(postRemaining / dimSize);
          postRemaining %= dimSize;
          inputIdx += coord * inputStrides[d]!;
        }

        const byte = aContiguous ? Number(a.data[aOff + inputIdx]!) : Number(a.iget(inputIdx));

        for (let bit = 0; bit < 8; bit++) {
          const axisIdx = packedIdx * 8 + bit;
          if (axisIdx >= unpackedAxisSize) break;

          // Calculate output linear index
          let outputIdx = 0;
          preRemaining = pre;
          for (let d = 0; d < axis; d++) {
            const dimSize =
              d < axis - 1 ? preAxisShape.slice(d + 1).reduce((acc, b) => acc * b, 1) : 1;
            const coord = Math.floor(preRemaining / dimSize);
            preRemaining %= dimSize;
            outputIdx += coord * outputStrides[d]!;
          }
          outputIdx += axisIdx * outputStrides[axis]!;
          postRemaining = post;
          for (let d = axis + 1; d < ndim; d++) {
            const dimSize =
              d < ndim - 1 ? postAxisShape.slice(d - axis).reduce((acc, b) => acc * b, 1) : 1;
            const coord = Math.floor(postRemaining / dimSize);
            postRemaining %= dimSize;
            outputIdx += coord * outputStrides[d]!;
          }

          if (bitorder === 'big') {
            resultData[outputIdx] = (byte >> (7 - bit)) & 1;
          } else {
            resultData[outputIdx] = (byte >> bit) & 1;
          }
        }
      }
    }
  }

  return result;
}

/**
 * Compute C-contiguous strides for a shape
 * @private
 */
function computeStrides(shape: number[]): number[] {
  const ndim = shape.length;
  const strides = new Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  return strides;
}

/**
 * Count the number of 1-bits in each element (population count).
 *
 * @param x - Input array (must be integer type)
 * @returns Array with population count for each element
 */
export function bitwise_count(x: ArrayStorage): ArrayStorage {
  const dtype = x.dtype;
  validateIntegerDType(dtype, 'bitwise_count');

  const shape = Array.from(x.shape);
  const data = x.data;
  const off = x.offset;
  const size = x.size;
  const contiguous = x.isCContiguous;

  // Output is always uint8 (max popcount is 64 for uint64)
  const result = ArrayStorage.zeros(shape, 'uint8');
  const resultData = result.data as Uint8Array;

  if (isBigIntDType(dtype)) {
    if (contiguous) {
      const typedData = data as BigInt64Array | BigUint64Array;
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = popcount64(typedData[i]!);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = popcount64(typedData[off + i]!);
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = popcount64(x.iget(i) as bigint);
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = popcount32(data[i] as number);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = popcount32(data[off + i] as number);
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = popcount32(Number(x.iget(i)));
      }
    }
  }

  return result;
}

/**
 * Count 1-bits in a 32-bit integer (handles signed values correctly)
 * @private
 */
function popcount32(n: number): number {
  // Convert to unsigned 32-bit representation
  n = n >>> 0;
  // Brian Kernighan's algorithm
  let count = 0;
  while (n !== 0) {
    n = n & (n - 1);
    count++;
  }
  return count;
}

/**
 * Count 1-bits in a 64-bit BigInt
 * @private
 */
function popcount64(n: bigint): number {
  // Handle negative numbers (convert to unsigned representation)
  if (n < 0n) {
    n = BigInt.asUintN(64, n);
  }
  let count = 0;
  while (n !== 0n) {
    n = n & (n - 1n);
    count++;
  }
  return count;
}

/**
 * Bitwise invert (alias for bitwise_not)
 *
 * @param x - Input array (must be integer type)
 * @returns Result storage with bitwise NOT values
 */
export function bitwise_invert(x: ArrayStorage): ArrayStorage {
  return bitwise_not(x);
}

/**
 * Bitwise left shift (alias for left_shift)
 *
 * @param x1 - Input array (must be integer type)
 * @param x2 - Shift amount (array or scalar)
 * @returns Result storage with left-shifted values
 */
export function bitwise_left_shift(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  return left_shift(x1, x2);
}

/**
 * Bitwise right shift (alias for right_shift)
 *
 * @param x1 - Input array (must be integer type)
 * @param x2 - Shift amount (array or scalar)
 * @returns Result storage with right-shifted values
 */
export function bitwise_right_shift(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  return right_shift(x1, x2);
}
