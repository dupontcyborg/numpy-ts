/**
 * Computation backend abstraction
 *
 * Internal module for element-wise and broadcast operations.
 * Provides a swappable backend for different computation strategies.
 *
 * @internal
 */

import { ArrayStorage } from '../core/storage';
import { promoteDTypes, isBigIntDType } from '../core/dtype';
import { Complex } from '../core/complex';

/**
 * Compute the broadcast shape of two arrays
 * Returns the shape that results from broadcasting a and b together
 * Throws if shapes are not compatible for broadcasting
 */
export function broadcastShapes(shapeA: readonly number[], shapeB: readonly number[]): number[] {
  const ndimA = shapeA.length;
  const ndimB = shapeB.length;
  const ndim = Math.max(ndimA, ndimB);
  const result = new Array(ndim);

  for (let i = 0; i < ndim; i++) {
    const dimA = i < ndim - ndimA ? 1 : shapeA[i - (ndim - ndimA)]!;
    const dimB = i < ndim - ndimB ? 1 : shapeB[i - (ndim - ndimB)]!;

    if (dimA === dimB) {
      result[i] = dimA;
    } else if (dimA === 1) {
      result[i] = dimB;
    } else if (dimB === 1) {
      result[i] = dimA;
    } else {
      throw new Error(
        `operands could not be broadcast together with shapes ${JSON.stringify(Array.from(shapeA))} ${JSON.stringify(Array.from(shapeB))}`
      );
    }
  }

  return result;
}

/**
 * Compute the strides for broadcasting an array to a target shape
 * Returns strides where dimensions that need broadcasting have stride 0
 */
function broadcastStrides(
  shape: readonly number[],
  strides: readonly number[],
  targetShape: readonly number[]
): number[] {
  const ndim = shape.length;
  const targetNdim = targetShape.length;
  const result = new Array(targetNdim).fill(0);

  // Align dimensions from the right
  for (let i = 0; i < ndim; i++) {
    const targetIdx = targetNdim - ndim + i;
    const dim = shape[i]!;
    const targetDim = targetShape[targetIdx]!;

    if (dim === targetDim) {
      // Same size, use original stride
      result[targetIdx] = strides[i]!;
    } else if (dim === 1) {
      // Broadcasting, stride is 0 (repeat along this dimension)
      result[targetIdx] = 0;
    } else {
      // This shouldn't happen if shapes were validated
      throw new Error('Invalid broadcast');
    }
  }

  return result;
}

/**
 * Create a broadcast view of an ArrayStorage
 * The returned storage shares data with the original but has different shape/strides
 */
function broadcastTo(storage: ArrayStorage, targetShape: readonly number[]): ArrayStorage {
  const broadcastedStrides = broadcastStrides(storage.shape, storage.strides, targetShape);
  return ArrayStorage.fromData(
    storage.data,
    Array.from(targetShape),
    storage.dtype,
    broadcastedStrides,
    storage.offset
  );
}

/**
 * Perform element-wise operation with broadcasting
 *
 * NOTE: This is the slow path for broadcasting/non-contiguous arrays.
 * Fast paths for contiguous arrays are implemented directly in ops/arithmetic.ts
 *
 * @param a - First array storage
 * @param b - Second array storage
 * @param op - Operation to perform (a, b) => result
 * @param opName - Name of operation (for special handling)
 * @returns Result storage
 */
export function elementwiseBinaryOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (a: number, b: number) => number,
  opName: string
): ArrayStorage {
  // Compute broadcast shape
  const outputShape = broadcastShapes(a.shape, b.shape);

  // Create broadcast views
  const aBroadcast = broadcastTo(a, outputShape);
  const bBroadcast = broadcastTo(b, outputShape);

  // Determine output dtype using NumPy promotion rules
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, resultDtype);
  const resultData = result.data;
  const size = result.size;

  if (isBigIntDType(resultDtype)) {
    // BigInt arithmetic - no precision loss
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const aRaw = aBroadcast.iget(i);
      const bRaw = bBroadcast.iget(i);

      // Convert to BigInt - handle case where value is already BigInt
      // Note: Complex values get their real part extracted
      const aNum = aRaw instanceof Complex ? aRaw.re : aRaw;
      const bNum = bRaw instanceof Complex ? bRaw.re : bRaw;
      const aVal = typeof aNum === 'bigint' ? aNum : BigInt(Math.round(aNum as number));
      const bVal = typeof bNum === 'bigint' ? bNum : BigInt(Math.round(bNum as number));

      // Use BigInt operations
      if (opName === 'add') {
        resultTyped[i] = aVal + bVal;
      } else if (opName === 'subtract') {
        resultTyped[i] = aVal - bVal;
      } else if (opName === 'multiply') {
        resultTyped[i] = aVal * bVal;
      } else if (opName === 'divide') {
        resultTyped[i] = aVal / bVal;
      } else {
        resultTyped[i] = BigInt(Math.round(op(Number(aVal), Number(bVal))));
      }
    }
  } else {
    // Regular numeric types (including float dtypes)
    // Need to convert BigInt values to Number if mixing dtypes
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    for (let i = 0; i < size; i++) {
      const aRaw = aBroadcast.iget(i);
      const bRaw = bBroadcast.iget(i);

      // Convert to Number if needed (handles BigInt → float promotion)
      const aVal = needsConversion && typeof aRaw === 'bigint' ? Number(aRaw) : Number(aRaw);
      const bVal = needsConversion && typeof bRaw === 'bigint' ? Number(bRaw) : Number(bRaw);

      resultData[i] = op(aVal, bVal);
    }
  }

  return result;
}

/**
 * Perform element-wise comparison with broadcasting
 * Returns boolean array (dtype: 'bool', stored as Uint8Array)
 */
export function elementwiseComparisonOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (a: number, b: number) => boolean
): ArrayStorage {
  // Compute broadcast shape
  const outputShape = broadcastShapes(a.shape, b.shape);

  // Create broadcast views
  const aBroadcast = broadcastTo(a, outputShape);
  const bBroadcast = broadcastTo(b, outputShape);

  // Get output shape
  const size = outputShape.reduce((a, b) => a * b, 1);

  // Create result array with bool dtype
  const resultData = new Uint8Array(size);

  // Check if we need to convert BigInt to Number for comparison
  const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

  // Perform element-wise comparison
  for (let i = 0; i < size; i++) {
    const aRaw = aBroadcast.iget(i);
    const bRaw = bBroadcast.iget(i);

    // Convert BigInt to Number if needed
    const aVal = needsConversion && typeof aRaw === 'bigint' ? Number(aRaw) : Number(aRaw);
    const bVal = needsConversion && typeof bRaw === 'bigint' ? Number(bRaw) : Number(bRaw);

    resultData[i] = op(aVal, bVal) ? 1 : 0;
  }

  return ArrayStorage.fromData(resultData, outputShape, 'bool');
}

/**
 * Perform element-wise unary operation
 *
 * @param a - Input array storage
 * @param op - Operation to perform (x) => result
 * @param preserveDtype - If true, preserve input dtype; if false, promote to float64 (default: true)
 * @returns Result storage
 */
export function elementwiseUnaryOp(
  a: ArrayStorage,
  op: (x: number) => number,
  preserveDtype = true
): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // Determine output dtype
  // Math operations like sqrt may need float output even for integer input
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  const resultDtype = preserveDtype ? dtype : isIntegerType ? 'float64' : dtype;

  // Create result storage
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;
  const inputData = a.data;

  if (isBigIntDType(dtype)) {
    // BigInt input - convert to Number for operation, then convert back if preserving dtype
    if (isBigIntDType(resultDtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        const val = Number(inputData[i]!);
        resultTyped[i] = BigInt(Math.round(op(val)));
      }
    } else {
      // BigInt input, float output
      for (let i = 0; i < size; i++) {
        resultData[i] = op(Number(inputData[i]!));
      }
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < size; i++) {
      resultData[i] = op(Number(inputData[i]!));
    }
  }

  return result;
}
