/**
 * Gradient and difference operations
 *
 * Pure functions for computing gradients and differences:
 * gradient, diff, ediff1d, cross
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType, isComplexDType, promoteDTypes } from '../core/dtype';

/**
 * Calculate the n-th discrete difference along the given axis.
 *
 * The first difference is given by out[i] = a[i+1] - a[i] along the given axis.
 * Higher differences are calculated by using diff recursively.
 *
 * @param a - Input array storage
 * @param n - Number of times values are differenced (default: 1)
 * @param axis - Axis along which to compute difference (default: -1, last axis)
 * @returns Result storage with differences
 */
export function diff(a: ArrayStorage, n: number = 1, axis: number = -1): ArrayStorage {
  if (n < 0) {
    throw new Error(`order must be non-negative but got ${n}`);
  }
  if (n === 0) {
    return a.copy();
  }

  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Check if we can take n differences
  if (shape[normalizedAxis]! < n + 1) {
    throw new Error(
      `diff requires at least ${n + 1} elements along axis ${axis}, but got ${shape[normalizedAxis]}`
    );
  }

  let result = a;

  for (let i = 0; i < n; i++) {
    result = diffOnce(result, normalizedAxis);
  }

  return result;
}

/**
 * Helper function to compute one difference operation
 * @private
 */
function diffOnce(a: ArrayStorage, axis: number): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const axisSize = shape[axis]!;

  // Result shape has one less element along the axis
  const resultShape = [...shape];
  resultShape[axis] = axisSize - 1;

  // Determine result dtype - always float64 for non-float types (except complex)
  const dtype = a.dtype;
  const isComplex = isComplexDType(dtype);
  const resultDtype = isBigIntDType(dtype) ? 'float64' : dtype;

  const result = ArrayStorage.zeros(resultShape, resultDtype);
  const resultData = result.data;

  // Calculate strides for manual iteration
  const strides = a.strides;

  // Calculate total size excluding the axis
  const resultSize = result.size;

  // Iterate over all elements in the result
  for (let resultIdx = 0; resultIdx < resultSize; resultIdx++) {
    // Convert flat index to multi-dimensional indices
    let remaining = resultIdx;
    const indices: number[] = new Array(ndim);

    for (let d = ndim - 1; d >= 0; d--) {
      indices[d] = remaining % resultShape[d]!;
      remaining = Math.floor(remaining / resultShape[d]!);
    }

    // Calculate source indices for a[i+1] and a[i] along the axis
    const idx1 = [...indices];
    const idx2 = [...indices];
    idx2[axis] = idx1[axis]! + 1;

    // Calculate flat indices in source array
    let flatIdx1 = 0;
    let flatIdx2 = 0;
    for (let d = 0; d < ndim; d++) {
      flatIdx1 += idx1[d]! * strides[d]!;
      flatIdx2 += idx2[d]! * strides[d]!;
    }

    // Compute difference
    if (isComplex) {
      // Complex: access interleaved data [re, im, re, im, ...]
      const complexData = a.data as Float64Array | Float32Array;
      const re1 = complexData[flatIdx1 * 2]!;
      const im1 = complexData[flatIdx1 * 2 + 1]!;
      const re2 = complexData[flatIdx2 * 2]!;
      const im2 = complexData[flatIdx2 * 2 + 1]!;
      (resultData as Float64Array | Float32Array)[resultIdx * 2] = re2 - re1;
      (resultData as Float64Array | Float32Array)[resultIdx * 2 + 1] = im2 - im1;
    } else {
      const val1 = isBigIntDType(dtype) ? Number(a.data[flatIdx1]!) : Number(a.data[flatIdx1]!);
      const val2 = isBigIntDType(dtype) ? Number(a.data[flatIdx2]!) : Number(a.data[flatIdx2]!);
      resultData[resultIdx] = val2 - val1;
    }
  }

  return result;
}

/**
 * The differences between consecutive elements of a flattened array.
 *
 * @param ary - Input array storage
 * @param to_end - Number(s) to append at the end of the returned differences
 * @param to_begin - Number(s) to prepend at the beginning of the returned differences
 * @returns Array of differences with optional prepend/append values
 */
export function ediff1d(
  ary: ArrayStorage,
  to_end: number[] | null = null,
  to_begin: number[] | null = null
): ArrayStorage {
  // Flatten the array
  const flatSize = ary.size;
  const dtype = ary.dtype;
  const isComplex = isComplexDType(dtype);
  const resultDtype = isBigIntDType(dtype) ? 'float64' : dtype;

  // Calculate result size
  const diffSize = Math.max(0, flatSize - 1);
  const beginSize = to_begin ? to_begin.length : 0;
  const endSize = to_end ? to_end.length : 0;
  const totalSize = beginSize + diffSize + endSize;

  const result = ArrayStorage.zeros([totalSize], resultDtype);
  const resultData = result.data;

  let idx = 0;

  // Add to_begin values (note: these are real numbers even for complex arrays)
  if (to_begin) {
    if (isComplex) {
      for (const val of to_begin) {
        (resultData as Float64Array | Float32Array)[idx * 2] = val;
        (resultData as Float64Array | Float32Array)[idx * 2 + 1] = 0;
        idx++;
      }
    } else {
      for (const val of to_begin) {
        resultData[idx++] = val;
      }
    }
  }

  // Calculate differences
  if (isComplex) {
    const complexData = ary.data as Float64Array | Float32Array;
    for (let i = 0; i < diffSize; i++) {
      const re1 = complexData[i * 2]!;
      const im1 = complexData[i * 2 + 1]!;
      const re2 = complexData[(i + 1) * 2]!;
      const im2 = complexData[(i + 1) * 2 + 1]!;
      (resultData as Float64Array | Float32Array)[idx * 2] = re2 - re1;
      (resultData as Float64Array | Float32Array)[idx * 2 + 1] = im2 - im1;
      idx++;
    }
  } else {
    for (let i = 0; i < diffSize; i++) {
      const val1 = isBigIntDType(dtype) ? Number(ary.iget(i)) : Number(ary.iget(i));
      const val2 = isBigIntDType(dtype) ? Number(ary.iget(i + 1)) : Number(ary.iget(i + 1));
      resultData[idx++] = val2 - val1;
    }
  }

  // Add to_end values (note: these are real numbers even for complex arrays)
  if (to_end) {
    if (isComplex) {
      for (const val of to_end) {
        (resultData as Float64Array | Float32Array)[idx * 2] = val;
        (resultData as Float64Array | Float32Array)[idx * 2 + 1] = 0;
        idx++;
      }
    } else {
      for (const val of to_end) {
        resultData[idx++] = val;
      }
    }
  }

  return result;
}

/**
 * Return the gradient of an N-dimensional array.
 *
 * The gradient is computed using second order accurate central differences in the interior
 * and first order accurate one-sided (forward or backwards) differences at the boundaries.
 *
 * @param f - Input array storage
 * @param varargs - Spacing between values (scalar or array per dimension)
 * @param axis - Axis or axes along which to compute gradient (default: all axes)
 * @returns Array of gradients (one per axis) or single gradient if one axis
 */
export function gradient(
  f: ArrayStorage,
  varargs: number | number[] = 1,
  axis: number | number[] | null = null
): ArrayStorage | ArrayStorage[] {
  const shape = Array.from(f.shape);
  const ndim = shape.length;

  // Determine which axes to compute gradient for
  let axes: number[];
  if (axis === null) {
    axes = Array.from({ length: ndim }, (_, i) => i);
  } else if (typeof axis === 'number') {
    const normalizedAxis = axis < 0 ? ndim + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }
    axes = [normalizedAxis];
  } else {
    axes = axis.map((ax) => {
      const normalized = ax < 0 ? ndim + ax : ax;
      if (normalized < 0 || normalized >= ndim) {
        throw new Error(`axis ${ax} is out of bounds for array of dimension ${ndim}`);
      }
      return normalized;
    });
  }

  // Determine spacing for each axis
  let spacings: number[];
  if (typeof varargs === 'number') {
    spacings = axes.map(() => varargs);
  } else {
    if (varargs.length !== axes.length) {
      throw new Error(`Number of spacings must match number of axes`);
    }
    spacings = varargs;
  }

  // Compute gradient for each axis
  const results: ArrayStorage[] = [];
  for (let i = 0; i < axes.length; i++) {
    results.push(gradientAlongAxis(f, axes[i]!, spacings[i]!));
  }

  // Return single result if single axis
  if (results.length === 1) {
    return results[0]!;
  }
  return results;
}

/**
 * Compute gradient along a single axis
 * @private
 */
function gradientAlongAxis(f: ArrayStorage, axis: number, spacing: number): ArrayStorage {
  const shape = Array.from(f.shape);
  const ndim = shape.length;
  const axisSize = shape[axis]!;

  if (axisSize < 2) {
    throw new Error(`Shape of array along axis ${axis} must be at least 2, but got ${axisSize}`);
  }

  // Result has same shape as input
  const dtype = f.dtype;
  const resultDtype = isBigIntDType(dtype)
    ? 'float64'
    : dtype === 'float32'
      ? 'float32'
      : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;
  const strides = f.strides;

  const h = spacing;
  const h2 = 2 * h;

  // Calculate total size
  const totalSize = f.size;

  // Iterate over all elements
  for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
    // Convert flat index to multi-dimensional indices
    let remaining = flatIdx;
    const indices: number[] = new Array(ndim);

    for (let d = ndim - 1; d >= 0; d--) {
      indices[d] = remaining % shape[d]!;
      remaining = Math.floor(remaining / shape[d]!);
    }

    const axisIdx = indices[axis]!;

    let gradientValue: number;

    if (axisIdx === 0) {
      // Forward difference at the beginning
      const idxPlus1 = [...indices];
      idxPlus1[axis] = 1;
      let flatIdxPlus1 = 0;
      for (let d = 0; d < ndim; d++) {
        flatIdxPlus1 += idxPlus1[d]! * strides[d]!;
      }

      const f0 = isBigIntDType(dtype) ? Number(f.data[flatIdx]!) : Number(f.data[flatIdx]!);
      const f1 = isBigIntDType(dtype)
        ? Number(f.data[flatIdxPlus1]!)
        : Number(f.data[flatIdxPlus1]!);

      gradientValue = (f1 - f0) / h;
    } else if (axisIdx === axisSize - 1) {
      // Backward difference at the end
      const idxMinus1 = [...indices];
      idxMinus1[axis] = axisSize - 2;
      let flatIdxMinus1 = 0;
      for (let d = 0; d < ndim; d++) {
        flatIdxMinus1 += idxMinus1[d]! * strides[d]!;
      }

      const fN = isBigIntDType(dtype) ? Number(f.data[flatIdx]!) : Number(f.data[flatIdx]!);
      const fNm1 = isBigIntDType(dtype)
        ? Number(f.data[flatIdxMinus1]!)
        : Number(f.data[flatIdxMinus1]!);

      gradientValue = (fN - fNm1) / h;
    } else {
      // Central difference in the interior
      const idxPlus1 = [...indices];
      const idxMinus1 = [...indices];
      idxPlus1[axis] = axisIdx + 1;
      idxMinus1[axis] = axisIdx - 1;

      let flatIdxPlus1 = 0;
      let flatIdxMinus1 = 0;
      for (let d = 0; d < ndim; d++) {
        flatIdxPlus1 += idxPlus1[d]! * strides[d]!;
        flatIdxMinus1 += idxMinus1[d]! * strides[d]!;
      }

      const fPlus = isBigIntDType(dtype)
        ? Number(f.data[flatIdxPlus1]!)
        : Number(f.data[flatIdxPlus1]!);
      const fMinus = isBigIntDType(dtype)
        ? Number(f.data[flatIdxMinus1]!)
        : Number(f.data[flatIdxMinus1]!);

      gradientValue = (fPlus - fMinus) / h2;
    }

    resultData[flatIdx] = gradientValue;
  }

  return result;
}

/**
 * Return the cross product of two (arrays of) vectors.
 *
 * The cross product of a and b in R^3 is a vector perpendicular to both a and b.
 *
 * @param a - First input array (components of first vector(s))
 * @param b - Second input array (components of second vector(s))
 * @param axisa - Axis of a that defines the vector(s) (default: -1)
 * @param axisb - Axis of b that defines the vector(s) (default: -1)
 * @param axisc - Axis of c containing the cross product vector(s) (default: -1)
 * @returns Cross product array
 */
export function cross(
  a: ArrayStorage,
  b: ArrayStorage,
  axisa: number = -1,
  axisb: number = -1,
  _axisc: number = -1
): ArrayStorage {
  const shapeA = Array.from(a.shape);
  const shapeB = Array.from(b.shape);
  const ndimA = shapeA.length;
  const ndimB = shapeB.length;

  // Normalize axes
  const normalizedAxisA = axisa < 0 ? ndimA + axisa : axisa;
  const normalizedAxisB = axisb < 0 ? ndimB + axisb : axisb;

  if (normalizedAxisA < 0 || normalizedAxisA >= ndimA) {
    throw new Error(`axisa ${axisa} is out of bounds for array of dimension ${ndimA}`);
  }
  if (normalizedAxisB < 0 || normalizedAxisB >= ndimB) {
    throw new Error(`axisb ${axisb} is out of bounds for array of dimension ${ndimB}`);
  }

  const sizeA = shapeA[normalizedAxisA]!;
  const sizeB = shapeB[normalizedAxisB]!;

  // Validate vector sizes (must be 2 or 3)
  if (sizeA !== 2 && sizeA !== 3) {
    throw new Error(
      `incompatible dimensions for cross product (dimension must be 2 or 3, got ${sizeA})`
    );
  }
  if (sizeB !== 2 && sizeB !== 3) {
    throw new Error(
      `incompatible dimensions for cross product (dimension must be 2 or 3, got ${sizeB})`
    );
  }

  // Determine result dtype
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // Handle simple 1D case for 3D vectors
  if (ndimA === 1 && ndimB === 1 && sizeA === 3 && sizeB === 3) {
    const result = ArrayStorage.zeros([3], resultDtype);
    const resultData = result.data;

    const a0 = Number(a.iget(0));
    const a1 = Number(a.iget(1));
    const a2 = Number(a.iget(2));
    const b0 = Number(b.iget(0));
    const b1 = Number(b.iget(1));
    const b2 = Number(b.iget(2));

    resultData[0] = a1 * b2 - a2 * b1;
    resultData[1] = a2 * b0 - a0 * b2;
    resultData[2] = a0 * b1 - a1 * b0;

    return result;
  }

  // Handle 1D case for 2D vectors (returns scalar)
  if (ndimA === 1 && ndimB === 1 && sizeA === 2 && sizeB === 2) {
    const result = ArrayStorage.zeros([], resultDtype);
    const a0 = Number(a.iget(0));
    const a1 = Number(a.iget(1));
    const b0 = Number(b.iget(0));
    const b1 = Number(b.iget(1));

    result.data[0] = a0 * b1 - a1 * b0;
    return result;
  }

  // Handle mixed 2D/3D case for 1D arrays
  if (ndimA === 1 && ndimB === 1) {
    // 2D x 3D or 3D x 2D
    if (sizeA === 2 && sizeB === 3) {
      // Treat 2D vector as [x, y, 0]
      const result = ArrayStorage.zeros([3], resultDtype);
      const resultData = result.data;

      const a0 = Number(a.iget(0));
      const a1 = Number(a.iget(1));
      const b0 = Number(b.iget(0));
      const b1 = Number(b.iget(1));
      const b2 = Number(b.iget(2));

      resultData[0] = a1 * b2;
      resultData[1] = -a0 * b2;
      resultData[2] = a0 * b1 - a1 * b0;

      return result;
    } else if (sizeA === 3 && sizeB === 2) {
      // 3D x 2D
      const result = ArrayStorage.zeros([3], resultDtype);
      const resultData = result.data;

      const a0 = Number(a.iget(0));
      const a1 = Number(a.iget(1));
      const a2 = Number(a.iget(2));
      const b0 = Number(b.iget(0));
      const b1 = Number(b.iget(1));

      resultData[0] = -a2 * b1;
      resultData[1] = a2 * b0;
      resultData[2] = a0 * b1 - a1 * b0;

      return result;
    }
  }

  // For higher-dimensional arrays, we need more complex logic
  // For now, handle 2D arrays where vectors are along the last axis
  if (ndimA === 2 && ndimB === 2 && normalizedAxisA === 1 && normalizedAxisB === 1) {
    const numVectors = shapeA[0]!;
    if (shapeB[0] !== numVectors) {
      throw new Error(`Shape mismatch: a has ${numVectors} vectors, b has ${shapeB[0]} vectors`);
    }

    if (sizeA === 3 && sizeB === 3) {
      const result = ArrayStorage.zeros([numVectors, 3], resultDtype);
      const resultData = result.data;

      for (let i = 0; i < numVectors; i++) {
        const a0 = Number(a.iget(i * 3));
        const a1 = Number(a.iget(i * 3 + 1));
        const a2 = Number(a.iget(i * 3 + 2));
        const b0 = Number(b.iget(i * 3));
        const b1 = Number(b.iget(i * 3 + 1));
        const b2 = Number(b.iget(i * 3 + 2));

        resultData[i * 3] = a1 * b2 - a2 * b1;
        resultData[i * 3 + 1] = a2 * b0 - a0 * b2;
        resultData[i * 3 + 2] = a0 * b1 - a1 * b0;
      }

      return result;
    }

    if (sizeA === 2 && sizeB === 2) {
      // Returns 1D array of scalar cross products
      const result = ArrayStorage.zeros([numVectors], resultDtype);
      const resultData = result.data;

      for (let i = 0; i < numVectors; i++) {
        const a0 = Number(a.iget(i * 2));
        const a1 = Number(a.iget(i * 2 + 1));
        const b0 = Number(b.iget(i * 2));
        const b1 = Number(b.iget(i * 2 + 1));

        resultData[i] = a0 * b1 - a1 * b0;
      }

      return result;
    }
  }

  throw new Error(
    `cross product not implemented for arrays with shapes ${JSON.stringify(shapeA)} and ${JSON.stringify(shapeB)}`
  );
}
