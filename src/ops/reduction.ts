/**
 * Reduction operations (sum, mean, max, min)
 *
 * Pure functions for reducing arrays along axes.
 * @module ops/reduction
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType, isComplexDType, throwIfComplex, type DType } from '../core/dtype';
import { outerIndexToMultiIndex, multiIndexToLinear } from '../internal/indexing';
import { Complex } from '../core/complex';

/**
 * Sum array elements over a given axis
 */
export function sum(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Sum all elements - return scalar (or Complex for complex arrays)
    if (isComplexDType(dtype)) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 0;
      let totalIm = 0;
      for (let i = 0; i < size; i++) {
        totalRe += complexData[i * 2]!;
        totalIm += complexData[i * 2 + 1]!;
      }
      return new Complex(totalRe, totalIm);
    } else if (isBigIntDType(dtype)) {
      const typedData = data as BigInt64Array | BigUint64Array;
      let total = BigInt(0);
      for (let i = 0; i < size; i++) {
        total += typedData[i]!;
      }
      return Number(total);
    } else {
      let total = 0;
      for (let i = 0; i < size; i++) {
        total += Number(data[i]!);
      }
      return total;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar - reuse scalar sum logic
    return sum(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isComplexDType(dtype)) {
    // Complex sum along axis
    const complexData = data as Float64Array | Float32Array;
    const resultComplex = resultData as Float64Array | Float32Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumRe = 0;
      let sumIm = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        // Physical index is 2x logical index for complex
        sumRe += complexData[linearIdx * 2]!;
        sumIm += complexData[linearIdx * 2 + 1]!;
      }
      // Output physical index is 2x output logical index
      resultComplex[outerIdx * 2] = sumRe;
      resultComplex[outerIdx * 2 + 1] = sumIm;
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = BigInt(0);
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        sumVal += typedData[linearIdx]!;
      }
      resultTyped[outerIdx] = sumVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        sumVal += Number(data[linearIdx]!);
      }
      resultData[outerIdx] = sumVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Compute the arithmetic mean along the specified axis
 * Note: mean() returns float64 for integer dtypes, matching NumPy behavior
 * For complex arrays, returns complex mean
 */
export function mean(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;

  if (axis === undefined) {
    const sumResult = sum(storage);
    if (sumResult instanceof Complex) {
      return new Complex(sumResult.re / storage.size, sumResult.im / storage.size);
    }
    return (sumResult as number) / storage.size;
  }

  // Normalize negative axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = shape.length + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= shape.length) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${shape.length}`);
  }

  const sumResult = sum(storage, axis, keepdims);
  if (typeof sumResult === 'number') {
    return sumResult / shape[normalizedAxis]!;
  }
  if (sumResult instanceof Complex) {
    return new Complex(
      sumResult.re / shape[normalizedAxis]!,
      sumResult.im / shape[normalizedAxis]!
    );
  }

  // Divide by the size of the reduced axis
  const divisor = shape[normalizedAxis]!;

  // For complex dtypes, mean stays complex
  // For integer dtypes, mean returns float64 (matching NumPy behavior)
  let resultDtype: DType = dtype;
  if (isComplexDType(dtype)) {
    // Complex mean stays complex
    resultDtype = dtype;
  } else if (isBigIntDType(dtype) || dtype.startsWith('int') || dtype.startsWith('uint')) {
    resultDtype = 'float64';
  }

  const result = ArrayStorage.zeros(Array.from(sumResult.shape), resultDtype);
  const resultData = result.data;
  const sumData = sumResult.data;

  if (isComplexDType(dtype)) {
    // Complex: divide both real and imaginary parts
    const sumComplex = sumData as Float64Array | Float32Array;
    const resultComplex = resultData as Float64Array | Float32Array;
    const size = sumResult.size;
    for (let i = 0; i < size; i++) {
      resultComplex[i * 2] = sumComplex[i * 2]! / divisor;
      resultComplex[i * 2 + 1] = sumComplex[i * 2 + 1]! / divisor;
    }
  } else if (isBigIntDType(dtype)) {
    // Convert BigInt sum results to float for mean
    const sumTyped = sumData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = Number(sumTyped[i]!) / divisor;
    }
  } else {
    for (let i = 0; i < resultData.length; i++) {
      resultData[i] = Number(sumData[i]!) / divisor;
    }
  }

  return result;
}

/**
 * Return the maximum along a given axis
 */
export function max(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  // Complex max uses lexicographic ordering (real first, then imaginary)
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      if (size === 0) {
        throw new Error('max of empty array');
      }

      let maxRe = complexData[0]!;
      let maxIm = complexData[1]!;

      for (let i = 1; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;

        // Check for NaN propagation
        if (isNaN(re) || isNaN(im)) {
          return new Complex(NaN, NaN);
        }

        // Lexicographic comparison: real first, then imaginary
        if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      // Check initial value for NaN
      if (isNaN(maxRe) || isNaN(maxIm)) {
        return new Complex(NaN, NaN);
      }
      return new Complex(maxRe, maxIm);
    }

    // Max along axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return max(storage) as Complex;
    }

    const result = ArrayStorage.zeros(outputShape, dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let maxRe = complexData[firstIdx * 2]!;
      let maxIm = complexData[firstIdx * 2 + 1]!;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;

        if (isNaN(re) || isNaN(im)) {
          maxRe = NaN;
          maxIm = NaN;
          break;
        }
        if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      resultData[outerIdx * 2] = maxRe;
      resultData[outerIdx * 2 + 1] = maxIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  if (axis === undefined) {
    // Max of all elements - return scalar
    if (size === 0) {
      throw new Error('max of empty array');
    }

    let maxVal = data[0]!;
    for (let i = 1; i < size; i++) {
      if (data[i]! > maxVal) {
        maxVal = data[i]!;
      }
    }
    return Number(maxVal);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return max(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let maxVal = typedData[firstIdx]!;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val > maxVal) {
          maxVal = val;
        }
      }
      resultTyped[outerIdx] = maxVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val > maxVal) {
          maxVal = val;
        }
      }
      resultData[outerIdx] = maxVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Product array elements over a given axis
 */
export function prod(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Product of all elements - return scalar (or Complex for complex arrays)
    if (isComplexDType(dtype)) {
      const complexData = data as Float64Array | Float32Array;
      let prodRe = 1;
      let prodIm = 0;
      for (let i = 0; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        const newRe = prodRe * re - prodIm * im;
        const newIm = prodRe * im + prodIm * re;
        prodRe = newRe;
        prodIm = newIm;
      }
      return new Complex(prodRe, prodIm);
    } else if (isBigIntDType(dtype)) {
      const typedData = data as BigInt64Array | BigUint64Array;
      let product = BigInt(1);
      for (let i = 0; i < size; i++) {
        product *= typedData[i]!;
      }
      return Number(product);
    } else {
      let product = 1;
      for (let i = 0; i < size; i++) {
        product *= Number(data[i]!);
      }
      return product;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar - reuse scalar prod logic
    return prod(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isComplexDType(dtype)) {
    // Complex product along axis
    const complexData = data as Float64Array | Float32Array;
    const resultComplex = resultData as Float64Array | Float32Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodRe = 1;
      let prodIm = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        const newRe = prodRe * re - prodIm * im;
        const newIm = prodRe * im + prodIm * re;
        prodRe = newRe;
        prodIm = newIm;
      }
      resultComplex[outerIdx * 2] = prodRe;
      resultComplex[outerIdx * 2 + 1] = prodIm;
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = BigInt(1);
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        prodVal *= typedData[linearIdx]!;
      }
      resultTyped[outerIdx] = prodVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = 1;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        prodVal *= Number(data[linearIdx]!);
      }
      resultData[outerIdx] = prodVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Return the minimum along a given axis
 */
export function min(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  // Complex min uses lexicographic ordering (real first, then imaginary)
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      if (size === 0) {
        throw new Error('min of empty array');
      }

      let minRe = complexData[0]!;
      let minIm = complexData[1]!;

      for (let i = 1; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;

        // Check for NaN propagation
        if (isNaN(re) || isNaN(im)) {
          return new Complex(NaN, NaN);
        }

        // Lexicographic comparison: real first, then imaginary
        if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      // Check initial value for NaN
      if (isNaN(minRe) || isNaN(minIm)) {
        return new Complex(NaN, NaN);
      }
      return new Complex(minRe, minIm);
    }

    // Min along axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return min(storage) as Complex;
    }

    const result = ArrayStorage.zeros(outputShape, dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let minRe = complexData[firstIdx * 2]!;
      let minIm = complexData[firstIdx * 2 + 1]!;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;

        if (isNaN(re) || isNaN(im)) {
          minRe = NaN;
          minIm = NaN;
          break;
        }
        if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      resultData[outerIdx * 2] = minRe;
      resultData[outerIdx * 2 + 1] = minIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  if (axis === undefined) {
    // Min of all elements - return scalar
    if (size === 0) {
      throw new Error('min of empty array');
    }

    let minVal = data[0]!;
    for (let i = 1; i < size; i++) {
      if (data[i]! < minVal) {
        minVal = data[i]!;
      }
    }
    return Number(minVal);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return min(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let minVal = typedData[firstIdx]!;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val < minVal) {
          minVal = val;
        }
      }
      resultTyped[outerIdx] = minVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val < minVal) {
          minVal = val;
        }
      }
      resultData[outerIdx] = minVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Return the indices of the minimum values along a given axis
 */
/**
 * Compare two complex numbers using lexicographic ordering (real first, then imaginary)
 * Returns -1 if a < b, 0 if a == b, 1 if a > b
 */
function complexCompare(aRe: number, aIm: number, bRe: number, bIm: number): number {
  if (aRe < bRe) return -1;
  if (aRe > bRe) return 1;
  // Real parts are equal, compare imaginary parts
  if (aIm < bIm) return -1;
  if (aIm > bIm) return 1;
  return 0;
}

export function argmin(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Argmin of all elements - return scalar index
    if (size === 0) {
      throw new Error('argmin of empty array');
    }

    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let minRe = complexData[0]!;
      let minIm = complexData[1]!;
      let minIdx = 0;
      for (let i = 1; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (complexCompare(re, im, minRe, minIm) < 0) {
          minRe = re;
          minIm = im;
          minIdx = i;
        }
      }
      return minIdx;
    }

    let minVal = data[0]!;
    let minIdx = 0;
    for (let i = 1; i < size; i++) {
      if (data[i]! < minVal) {
        minVal = data[i]!;
        minIdx = i;
      }
    }
    return minIdx;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return argmin(storage);
  }

  // Create result storage with int32 dtype (indices are always integers)
  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let minRe = complexData[firstIdx * 2]!;
      let minIm = complexData[firstIdx * 2 + 1]!;
      let minAxisIdx = 0;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (complexCompare(re, im, minRe, minIm) < 0) {
          minRe = re;
          minIm = im;
          minAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = minAxisIdx;
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let minVal = typedData[firstIdx]!;
      let minAxisIdx = 0;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val < minVal) {
          minVal = val;
          minAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = minAxisIdx;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      let minAxisIdx = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val < minVal) {
          minVal = val;
          minAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = minAxisIdx;
    }
  }

  return result;
}

/**
 * Return the indices of the maximum values along a given axis
 */
export function argmax(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Argmax of all elements - return scalar index
    if (size === 0) {
      throw new Error('argmax of empty array');
    }

    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let maxRe = complexData[0]!;
      let maxIm = complexData[1]!;
      let maxIdx = 0;
      for (let i = 1; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (complexCompare(re, im, maxRe, maxIm) > 0) {
          maxRe = re;
          maxIm = im;
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    let maxVal = data[0]!;
    let maxIdx = 0;
    for (let i = 1; i < size; i++) {
      if (data[i]! > maxVal) {
        maxVal = data[i]!;
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return argmax(storage);
  }

  // Create result storage with int32 dtype (indices are always integers)
  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let maxRe = complexData[firstIdx * 2]!;
      let maxIm = complexData[firstIdx * 2 + 1]!;
      let maxAxisIdx = 0;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (complexCompare(re, im, maxRe, maxIm) > 0) {
          maxRe = re;
          maxIm = im;
          maxAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Initialize with first value along axis
      const firstIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, 0, shape);
      const firstIdx = multiIndexToLinear(firstIndices, shape);
      let maxVal = typedData[firstIdx]!;
      let maxAxisIdx = 0;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = typedData[linearIdx]!;
        if (val > maxVal) {
          maxVal = val;
          maxAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      let maxAxisIdx = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const val = Number(data[linearIdx]!);
        if (val > maxVal) {
          maxVal = val;
          maxAxisIdx = axisIdx;
        }
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  }

  return result;
}

/**
 * Compute the variance along the specified axis
 * @param storage - Input array storage
 * @param axis - Axis along which to compute variance
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - Keep dimensions (default: false)
 *
 * For complex arrays: Var(X) = E[|X - E[X]|²] where |z|² = re² + im²
 * Returns real values (float64) for both real and complex input
 */
export function variance(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  // Compute mean
  const meanResult = mean(storage, axis, keepdims);

  if (axis === undefined) {
    // Variance of all elements - return scalar
    if (isComplexDType(dtype)) {
      // For complex: Var(X) = E[|X - μ|²]
      const complexData = data as Float64Array | Float32Array;
      const meanComplex = meanResult as Complex;
      let sumSqDiff = 0;

      for (let i = 0; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        // |z - μ|² = (re - μ.re)² + (im - μ.im)²
        const diffRe = re - meanComplex.re;
        const diffIm = im - meanComplex.im;
        sumSqDiff += diffRe * diffRe + diffIm * diffIm;
      }

      return sumSqDiff / (size - ddof);
    }

    const meanVal = meanResult as number;
    let sumSqDiff = 0;

    for (let i = 0; i < size; i++) {
      const diff = Number(data[i]!) - meanVal;
      sumSqDiff += diff * diff;
    }

    return sumSqDiff / (size - ddof);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const meanArray = meanResult as ArrayStorage;
  const meanData = meanArray.data;

  // Compute output shape (same as mean's output shape)
  const outputShape = keepdims
    ? meanArray.shape
    : Array.from(shape).filter((_, i) => i !== normalizedAxis);

  // Result is always float64 for variance (even for complex input)
  const result = ArrayStorage.zeros(Array.from(outputShape), 'float64');
  const resultData = result.data;

  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  if (isComplexDType(dtype)) {
    // Complex variance along axis: Var(X) = E[|X - μ|²]
    const complexData = data as Float64Array | Float32Array;
    const meanComplex = meanData as Float64Array | Float32Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumSqDiff = 0;
      const meanRe = meanComplex[outerIdx * 2]!;
      const meanIm = meanComplex[outerIdx * 2 + 1]!;

      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        // |z - μ|² = (re - μ.re)² + (im - μ.im)²
        const diffRe = re - meanRe;
        const diffIm = im - meanIm;
        sumSqDiff += diffRe * diffRe + diffIm * diffIm;
      }

      resultData[outerIdx] = sumSqDiff / (axisSize - ddof);
    }
  } else {
    // Real variance for each position
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumSqDiff = 0;
      const meanVal = Number(meanData[outerIdx]!);

      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const diff = Number(data[linearIdx]!) - meanVal;
        sumSqDiff += diff * diff;
      }

      resultData[outerIdx] = sumSqDiff / (axisSize - ddof);
    }
  }

  return result;
}

/**
 * Compute the standard deviation along the specified axis
 * @param storage - Input array storage
 * @param axis - Axis along which to compute std
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - Keep dimensions (default: false)
 *
 * For complex arrays: returns sqrt(Var(X)) where Var(X) = E[|X - E[X]|²]
 * Returns real values (float64) for both real and complex input
 */
export function std(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  // variance() handles complex arrays - returns real values
  const varResult = variance(storage, axis, ddof, keepdims);

  if (typeof varResult === 'number') {
    return Math.sqrt(varResult);
  }

  // Apply sqrt element-wise
  const result = ArrayStorage.zeros(Array.from(varResult.shape), 'float64');
  const varData = varResult.data;
  const resultData = result.data;

  for (let i = 0; i < varData.length; i++) {
    resultData[i] = Math.sqrt(Number(varData[i]!));
  }

  return result;
}

/**
 * Test whether all array elements along a given axis evaluate to True
 */
export function all(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | boolean {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Test all elements
    for (let i = 0; i < size; i++) {
      if (!data[i]) {
        return false;
      }
    }
    return true;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return all(storage);
  }

  // Create result storage with bool dtype
  const result = ArrayStorage.zeros(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let allTrue = true;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      if (!data[linearIdx]) {
        allTrue = false;
        break;
      }
    }
    resultData[outerIdx] = allTrue ? 1 : 0;
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'bool');
  }

  return result;
}

/**
 * Test whether any array elements along a given axis evaluate to True
 */
export function any(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | boolean {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  if (axis === undefined) {
    // Test all elements
    for (let i = 0; i < size; i++) {
      if (data[i]) {
        return true;
      }
    }
    return false;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return any(storage);
  }

  // Create result storage with bool dtype
  const result = ArrayStorage.zeros(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let anyTrue = false;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      if (data[linearIdx]) {
        anyTrue = true;
        break;
      }
    }
    resultData[outerIdx] = anyTrue ? 1 : 0;
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'bool');
  }

  return result;
}

/**
 * Return cumulative sum of elements along a given axis
 * For complex arrays: returns complex cumulative sum
 */
export function cumsum(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex cumsum
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumsum
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let sumRe = 0;
      let sumIm = 0;
      for (let i = 0; i < size; i++) {
        sumRe += complexData[i * 2]!;
        sumIm += complexData[i * 2 + 1]!;
        resultData[i * 2] = sumRe;
        resultData[i * 2 + 1] = sumIm;
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }

    // Perform cumsum along axis
    const totalSize = storage.size;
    const axisStride = strides[normalizedAxis]!;

    for (let i = 0; i < totalSize; i++) {
      const axisPos = Math.floor(i / axisStride) % axisSize;

      if (axisPos === 0) {
        resultData[i * 2] = complexData[i * 2]!;
        resultData[i * 2 + 1] = complexData[i * 2 + 1]!;
      } else {
        resultData[i * 2] = resultData[(i - axisStride) * 2]! + complexData[i * 2]!;
        resultData[i * 2 + 1] = resultData[(i - axisStride) * 2 + 1]! + complexData[i * 2 + 1]!;
      }
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Flatten and cumsum
    const size = storage.size;
    const resultData = new Float64Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      sum += Number(data[i]);
      resultData[i] = sum;
    }
    return ArrayStorage.fromData(resultData, [size], 'float64');
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape
  const resultData = new Float64Array(storage.size);
  const axisSize = shape[normalizedAxis]!;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Perform cumsum along axis
  const totalSize = storage.size;
  const axisStride = strides[normalizedAxis]!;

  for (let i = 0; i < totalSize; i++) {
    // Determine position along axis
    const axisPos = Math.floor(i / axisStride) % axisSize;

    if (axisPos === 0) {
      resultData[i] = Number(data[i]);
    } else {
      // Add previous element along axis
      resultData[i] = resultData[i - axisStride]! + Number(data[i]);
    }
  }

  return ArrayStorage.fromData(resultData, [...shape], 'float64');
}

/**
 * Return cumulative product of elements along a given axis
 * For complex arrays: returns complex cumulative product
 * Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 */
export function cumprod(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex cumprod
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumprod
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let prodRe = 1;
      let prodIm = 0;
      for (let i = 0; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        const newRe = prodRe * re - prodIm * im;
        const newIm = prodRe * im + prodIm * re;
        prodRe = newRe;
        prodIm = newIm;
        resultData[i * 2] = prodRe;
        resultData[i * 2 + 1] = prodIm;
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }

    // Perform cumprod along axis
    const totalSize = storage.size;
    const axisStride = strides[normalizedAxis]!;

    for (let i = 0; i < totalSize; i++) {
      const axisPos = Math.floor(i / axisStride) % axisSize;

      if (axisPos === 0) {
        resultData[i * 2] = complexData[i * 2]!;
        resultData[i * 2 + 1] = complexData[i * 2 + 1]!;
      } else {
        // Multiply by previous element: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        const prevRe = resultData[(i - axisStride) * 2]!;
        const prevIm = resultData[(i - axisStride) * 2 + 1]!;
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        resultData[i * 2] = prevRe * re - prevIm * im;
        resultData[i * 2 + 1] = prevRe * im + prevIm * re;
      }
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Flatten and cumprod
    const size = storage.size;
    const resultData = new Float64Array(size);
    let prod = 1;
    for (let i = 0; i < size; i++) {
      prod *= Number(data[i]);
      resultData[i] = prod;
    }
    return ArrayStorage.fromData(resultData, [size], 'float64');
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape
  const resultData = new Float64Array(storage.size);
  const axisSize = shape[normalizedAxis]!;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Perform cumprod along axis
  const totalSize = storage.size;
  const axisStride = strides[normalizedAxis]!;

  for (let i = 0; i < totalSize; i++) {
    // Determine position along axis
    const axisPos = Math.floor(i / axisStride) % axisSize;

    if (axisPos === 0) {
      resultData[i] = Number(data[i]);
    } else {
      // Multiply by previous element along axis
      resultData[i] = resultData[i - axisStride]! * Number(data[i]);
    }
  }

  return ArrayStorage.fromData(resultData, [...shape], 'float64');
}

/**
 * Peak to peak (maximum - minimum) value along a given axis
 */
export function ptp(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;

  // Complex ptp: max - min using lexicographic ordering
  if (isComplexDType(dtype)) {
    const maxResult = max(storage, axis, keepdims) as Complex | ArrayStorage;
    const minResult = min(storage, axis, keepdims) as Complex | ArrayStorage;

    if (maxResult instanceof Complex && minResult instanceof Complex) {
      return new Complex(maxResult.re - minResult.re, maxResult.im - minResult.im);
    }

    // Both are arrays, subtract element-wise
    const maxStorage = maxResult as ArrayStorage;
    const minStorage = minResult as ArrayStorage;
    const maxData = maxStorage.data as Float64Array | Float32Array;
    const minData = minStorage.data as Float64Array | Float32Array;
    const resultData = new Float64Array(maxStorage.size * 2);

    for (let i = 0; i < maxStorage.size; i++) {
      resultData[i * 2] = maxData[i * 2]! - minData[i * 2]!;
      resultData[i * 2 + 1] = maxData[i * 2 + 1]! - minData[i * 2 + 1]!;
    }

    return ArrayStorage.fromData(resultData, [...maxStorage.shape], dtype);
  }

  const maxResult = max(storage, axis, keepdims);
  const minResult = min(storage, axis, keepdims);

  if (typeof maxResult === 'number' && typeof minResult === 'number') {
    return maxResult - minResult;
  }

  // Both are arrays, subtract element-wise
  const maxStorage = maxResult as ArrayStorage;
  const minStorage = minResult as ArrayStorage;
  const maxData = maxStorage.data;
  const minData = minStorage.data;
  const resultData = new Float64Array(maxStorage.size);

  for (let i = 0; i < maxStorage.size; i++) {
    resultData[i] = Number(maxData[i]) - Number(minData[i]);
  }

  return ArrayStorage.fromData(resultData, [...maxStorage.shape], 'float64');
}

/**
 * Compute the median along the specified axis
 */
export function median(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  return quantile(storage, 0.5, axis, keepdims);
}

/**
 * Compute the q-th percentile of data along specified axis
 */
export function percentile(
  storage: ArrayStorage,
  q: number,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  return quantile(storage, q / 100, axis, keepdims);
}

/**
 * Compute the q-th quantile of data along specified axis
 */
export function quantile(
  storage: ArrayStorage,
  q: number,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  throwIfComplex(storage.dtype, 'quantile', 'Complex numbers are not orderable.');
  if (q < 0 || q > 1) {
    throw new Error('Quantile must be between 0 and 1');
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (axis === undefined) {
    // Compute quantile over all elements
    const values: number[] = [];
    for (let i = 0; i < storage.size; i++) {
      values.push(Number(data[i]));
    }
    values.sort((a, b) => a - b);

    const n = values.length;
    const idx = q * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);

    if (lower === upper) {
      return values[lower]!;
    }

    // Linear interpolation
    const frac = idx - lower;
    return values[lower]! * (1 - frac) + values[upper]! * frac;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return quantile(storage, q);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    // Collect values along axis
    const values: number[] = [];
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      values.push(Number(data[linearIdx]));
    }
    values.sort((a, b) => a - b);

    const n = values.length;
    const idx = q * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);

    if (lower === upper) {
      resultData[outerIdx] = values[lower]!;
    } else {
      // Linear interpolation
      const frac = idx - lower;
      resultData[outerIdx] = values[lower]! * (1 - frac) + values[upper]! * frac;
    }
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute the weighted average along the specified axis
 */
export function average(
  storage: ArrayStorage,
  axis?: number,
  weights?: ArrayStorage,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (weights === undefined) {
    // Unweighted average is just mean
    return mean(storage, axis, keepdims);
  }

  if (isComplexDType(dtype)) {
    // Complex weighted average: sum(w_i * z_i) / sum(w_i)
    const complexData = data as Float64Array | Float32Array;
    const weightData = weights.data;

    if (axis === undefined) {
      // Compute weighted average over all elements
      let sumRe = 0;
      let sumIm = 0;
      let sumWeights = 0;

      for (let i = 0; i < storage.size; i++) {
        const w = Number(weightData[i % weights.size]);
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        sumRe += re * w;
        sumIm += im * w;
        sumWeights += w;
      }

      if (sumWeights === 0) {
        return new Complex(NaN, NaN);
      }
      return new Complex(sumRe / sumWeights, sumIm / sumWeights);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Compute output shape
    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return average(storage, undefined, weights);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.zeros(outputShape, dtype);
    const resultData = result.data as Float64Array | Float32Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumRe = 0;
      let sumIm = 0;
      let sumWeights = 0;

      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const w = Number(weightData[axisIdx % weights.size]);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        sumRe += re * w;
        sumIm += im * w;
        sumWeights += w;
      }

      if (sumWeights === 0) {
        resultData[outerIdx * 2] = NaN;
        resultData[outerIdx * 2 + 1] = NaN;
      } else {
        resultData[outerIdx * 2] = sumRe / sumWeights;
        resultData[outerIdx * 2 + 1] = sumIm / sumWeights;
      }
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Compute weighted average over all elements
    let sumWeightedValues = 0;
    let sumWeights = 0;
    const weightData = weights.data;

    for (let i = 0; i < storage.size; i++) {
      const w = Number(weightData[i % weights.size]);
      sumWeightedValues += Number(data[i]) * w;
      sumWeights += w;
    }

    return sumWeights === 0 ? NaN : sumWeightedValues / sumWeights;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return average(storage, undefined, weights);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const weightData = weights.data;
  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let sumWeightedValues = 0;
    let sumWeights = 0;

    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const w = Number(weightData[axisIdx % weights.size]);
      sumWeightedValues += Number(data[linearIdx]) * w;
      sumWeights += w;
    }

    resultData[outerIdx] = sumWeights === 0 ? NaN : sumWeightedValues / sumWeights;
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

// ============================================================================
// NaN-aware reduction functions
// ============================================================================

/**
 * Return sum of elements, treating NaNs as zero
 */
/**
 * Check if a complex number is NaN (either part is NaN)
 */
function complexIsNaN(re: number, im: number): boolean {
  return isNaN(re) || isNaN(im);
}

export function nansum(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (axis === undefined) {
    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 0;
      let totalIm = 0;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
        }
      }
      return new Complex(totalRe, totalIm);
    }

    let total = 0;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        total += val;
      }
    }
    return total;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nansum(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    const resultData = new Float64Array(outerSize * 2);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let totalRe = 0;
      let totalIm = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
        }
      }
      resultData[outerIdx * 2] = totalRe;
      resultData[outerIdx * 2 + 1] = totalIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return ArrayStorage.fromData(resultData, outputShape, dtype);
  }

  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let total = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val)) {
        total += val;
      }
    }
    resultData[outerIdx] = total;
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Return product of elements, treating NaNs as one
 */
export function nanprod(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (axis === undefined) {
    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 1;
      let totalIm = 0;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
          const newRe = totalRe * re - totalIm * im;
          const newIm = totalRe * im + totalIm * re;
          totalRe = newRe;
          totalIm = newIm;
        }
      }
      return new Complex(totalRe, totalIm);
    }

    let total = 1;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        total *= val;
      }
    }
    return total;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanprod(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    const resultData = new Float64Array(outerSize * 2);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let totalRe = 1;
      let totalIm = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          const newRe = totalRe * re - totalIm * im;
          const newIm = totalRe * im + totalIm * re;
          totalRe = newRe;
          totalIm = newIm;
        }
      }
      resultData[outerIdx * 2] = totalRe;
      resultData[outerIdx * 2 + 1] = totalIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return ArrayStorage.fromData(resultData, outputShape, dtype);
  }

  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let total = 1;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val)) {
        total *= val;
      }
    }
    resultData[outerIdx] = total;
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute mean ignoring NaN values
 */
export function nanmean(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (axis === undefined) {
    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
          count++;
        }
      }
      return count === 0 ? new Complex(NaN, NaN) : new Complex(totalRe / count, totalIm / count);
    }

    let total = 0;
    let count = 0;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        total += val;
        count++;
      }
    }
    return count === 0 ? NaN : total / count;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanmean(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    const resultData = new Float64Array(outerSize * 2);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
          count++;
        }
      }
      if (count === 0) {
        resultData[outerIdx * 2] = NaN;
        resultData[outerIdx * 2 + 1] = NaN;
      } else {
        resultData[outerIdx * 2] = totalRe / count;
        resultData[outerIdx * 2 + 1] = totalIm / count;
      }
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return ArrayStorage.fromData(resultData, outputShape, dtype);
  }

  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let total = 0;
    let count = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val)) {
        total += val;
        count++;
      }
    }
    resultData[outerIdx] = count === 0 ? NaN : total / count;
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute variance ignoring NaN values
 */
export function nanvar(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex nanvar: Var(X) = E[|X - μ|²] where |z|² = re² + im²
    // Returns real values (float64)
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      // First pass: compute mean ignoring NaN
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
          count++;
        }
      }
      if (count - ddof <= 0) return NaN;
      const meanRe = totalRe / count;
      const meanIm = totalIm / count;

      // Second pass: compute sum of squared deviations
      let sumSq = 0;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          // |z - μ|² = (re - μ.re)² + (im - μ.im)²
          const diffRe = re - meanRe;
          const diffIm = im - meanIm;
          sumSq += diffRe * diffRe + diffIm * diffIm;
        }
      }
      return sumSq / (count - ddof);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanvar(storage, undefined, ddof);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const resultData = new Float64Array(outerSize);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // First pass: compute mean ignoring NaN
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
          count++;
        }
      }

      if (count - ddof <= 0) {
        resultData[outerIdx] = NaN;
        continue;
      }

      const meanRe = totalRe / count;
      const meanIm = totalIm / count;

      // Second pass: compute sum of squared deviations
      let sumSq = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          const diffRe = re - meanRe;
          const diffIm = im - meanIm;
          sumSq += diffRe * diffRe + diffIm * diffIm;
        }
      }
      resultData[outerIdx] = sumSq / (count - ddof);
    }

    const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // First pass: compute mean
    let total = 0;
    let count = 0;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        total += val;
        count++;
      }
    }
    if (count - ddof <= 0) return NaN;
    const meanVal = total / count;

    // Second pass: compute sum of squared deviations
    let sumSq = 0;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        sumSq += (val - meanVal) ** 2;
      }
    }
    return sumSq / (count - ddof);
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanvar(storage, undefined, ddof);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    // First pass: compute mean
    let total = 0;
    let count = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val)) {
        total += val;
        count++;
      }
    }

    if (count - ddof <= 0) {
      resultData[outerIdx] = NaN;
      continue;
    }

    const meanVal = total / count;

    // Second pass: compute sum of squared deviations
    let sumSq = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val)) {
        sumSq += (val - meanVal) ** 2;
      }
    }
    resultData[outerIdx] = sumSq / (count - ddof);
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute standard deviation ignoring NaN values
 * For complex arrays: returns sqrt of variance (always real values)
 */
export function nanstd(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  // nanvar handles complex arrays and returns real values
  const varResult = nanvar(storage, axis, ddof, keepdims);
  if (typeof varResult === 'number') {
    return Math.sqrt(varResult);
  }
  const varStorage = varResult as ArrayStorage;
  const resultData = new Float64Array(varStorage.size);
  for (let i = 0; i < varStorage.size; i++) {
    resultData[i] = Math.sqrt(Number(varStorage.data[i]));
  }
  return ArrayStorage.fromData(resultData, [...varStorage.shape], 'float64');
}

/**
 * Return minimum ignoring NaN values
 */
export function nanmin(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  // Complex nanmin uses lexicographic ordering, skipping NaN values
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let minRe = Infinity;
      let minIm = Infinity;
      let foundNonNaN = false;

      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;

        // Skip NaN values
        if (isNaN(re) || isNaN(im)) {
          continue;
        }

        if (!foundNonNaN) {
          minRe = re;
          minIm = im;
          foundNonNaN = true;
        } else if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      return foundNonNaN ? new Complex(minRe, minIm) : new Complex(NaN, NaN);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanmin(storage) as Complex;
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const resultData = new Float64Array(outerSize * 2);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minRe = Infinity;
      let minIm = Infinity;
      let foundNonNaN = false;

      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;

        if (isNaN(re) || isNaN(im)) {
          continue;
        }

        if (!foundNonNaN) {
          minRe = re;
          minIm = im;
          foundNonNaN = true;
        } else if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      resultData[outerIdx * 2] = foundNonNaN ? minRe : NaN;
      resultData[outerIdx * 2 + 1] = foundNonNaN ? minIm : NaN;
    }

    const result = ArrayStorage.fromData(resultData, outputShape, dtype);

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }

    return result;
  }

  if (axis === undefined) {
    let minVal = Infinity;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val) && val < minVal) {
        minVal = val;
      }
    }
    return minVal === Infinity ? NaN : minVal;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanmin(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let minVal = Infinity;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val) && val < minVal) {
        minVal = val;
      }
    }
    resultData[outerIdx] = minVal === Infinity ? NaN : minVal;
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Return maximum ignoring NaN values
 */
export function nanmax(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  // Complex nanmax uses lexicographic ordering, skipping NaN values
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let foundNonNaN = false;

      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;

        // Skip NaN values
        if (isNaN(re) || isNaN(im)) {
          continue;
        }

        if (!foundNonNaN) {
          maxRe = re;
          maxIm = im;
          foundNonNaN = true;
        } else if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      return foundNonNaN ? new Complex(maxRe, maxIm) : new Complex(NaN, NaN);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanmax(storage) as Complex;
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const resultData = new Float64Array(outerSize * 2);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let foundNonNaN = false;

      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;

        if (isNaN(re) || isNaN(im)) {
          continue;
        }

        if (!foundNonNaN) {
          maxRe = re;
          maxIm = im;
          foundNonNaN = true;
        } else if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      resultData[outerIdx * 2] = foundNonNaN ? maxRe : NaN;
      resultData[outerIdx * 2 + 1] = foundNonNaN ? maxIm : NaN;
    }

    const result = ArrayStorage.fromData(resultData, outputShape, dtype);

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }

    return result;
  }

  if (axis === undefined) {
    let maxVal = -Infinity;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val) && val > maxVal) {
        maxVal = val;
      }
    }
    return maxVal === -Infinity ? NaN : maxVal;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanmax(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let maxVal = -Infinity;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val) && val > maxVal) {
        maxVal = val;
      }
    }
    resultData[outerIdx] = maxVal === -Infinity ? NaN : maxVal;
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Return indices of minimum value, ignoring NaNs
 */
export function nanargmin(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex nanargmin using lexicographic ordering, skipping NaN values
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let minRe = Infinity;
      let minIm = Infinity;
      let minIdx = -1;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im) && complexCompare(re, im, minRe, minIm) < 0) {
          minRe = re;
          minIm = im;
          minIdx = i;
        }
      }
      return minIdx;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanargmin(storage);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const resultData = new Int32Array(outerSize);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minRe = Infinity;
      let minIm = Infinity;
      let minIdx = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im) && complexCompare(re, im, minRe, minIm) < 0) {
          minRe = re;
          minIm = im;
          minIdx = axisIdx;
        }
      }
      resultData[outerIdx] = minIdx;
    }

    return ArrayStorage.fromData(resultData, outputShape, 'int32');
  }

  // Non-complex path
  if (axis === undefined) {
    let minVal = Infinity;
    let minIdx = -1;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val) && val < minVal) {
        minVal = val;
        minIdx = i;
      }
    }
    return minIdx;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanargmin(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Int32Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let minVal = Infinity;
    let minIdx = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val) && val < minVal) {
        minVal = val;
        minIdx = axisIdx;
      }
    }
    resultData[outerIdx] = minIdx;
  }

  return ArrayStorage.fromData(resultData, outputShape, 'int32');
}

/**
 * Return indices of maximum value, ignoring NaNs
 * For complex arrays: uses lexicographic ordering (real first, then imaginary)
 */
export function nanargmax(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex nanargmax using lexicographic ordering, skipping NaN values
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let maxIdx = -1;
      for (let i = 0; i < storage.size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im) && complexCompare(re, im, maxRe, maxIm) > 0) {
          maxRe = re;
          maxIm = im;
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanargmax(storage);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const resultData = new Int32Array(outerSize);

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let maxIdx = 0;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        const re = complexData[linearIdx * 2]!;
        const im = complexData[linearIdx * 2 + 1]!;
        if (!complexIsNaN(re, im) && complexCompare(re, im, maxRe, maxIm) > 0) {
          maxRe = re;
          maxIm = im;
          maxIdx = axisIdx;
        }
      }
      resultData[outerIdx] = maxIdx;
    }

    return ArrayStorage.fromData(resultData, outputShape, 'int32');
  }

  // Non-complex path
  if (axis === undefined) {
    let maxVal = -Infinity;
    let maxIdx = -1;
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val) && val > maxVal) {
        maxVal = val;
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanargmax(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Int32Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val) && val > maxVal) {
        maxVal = val;
        maxIdx = axisIdx;
      }
    }
    resultData[outerIdx] = maxIdx;
  }

  return ArrayStorage.fromData(resultData, outputShape, 'int32');
}

/**
 * Return cumulative sum, treating NaNs as zero
 */
export function nancumsum(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex nancumsum - treat NaN values as 0
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumsum
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let sumRe = 0;
      let sumIm = 0;
      for (let i = 0; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          sumRe += re;
          sumIm += im;
        }
        resultData[i * 2] = sumRe;
        resultData[i * 2 + 1] = sumIm;
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }

    // Perform cumsum along axis
    const totalSize = storage.size;
    const axisStride = strides[normalizedAxis]!;

    for (let i = 0; i < totalSize; i++) {
      const re = complexData[i * 2]!;
      const im = complexData[i * 2 + 1]!;
      const axisPos = Math.floor(i / axisStride) % axisSize;
      const isNan = complexIsNaN(re, im);

      if (axisPos === 0) {
        resultData[i * 2] = isNan ? 0 : re;
        resultData[i * 2 + 1] = isNan ? 0 : im;
      } else {
        resultData[i * 2] = resultData[(i - axisStride) * 2]! + (isNan ? 0 : re);
        resultData[i * 2 + 1] = resultData[(i - axisStride) * 2 + 1]! + (isNan ? 0 : im);
      }
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Flatten and cumsum
    const size = storage.size;
    const resultData = new Float64Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        sum += val;
      }
      resultData[i] = sum;
    }
    return ArrayStorage.fromData(resultData, [size], 'float64');
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape
  const resultData = new Float64Array(storage.size);
  const axisSize = shape[normalizedAxis]!;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Perform cumsum along axis
  const totalSize = storage.size;
  const axisStride = strides[normalizedAxis]!;

  for (let i = 0; i < totalSize; i++) {
    const val = Number(data[i]);
    const axisPos = Math.floor(i / axisStride) % axisSize;

    if (axisPos === 0) {
      resultData[i] = isNaN(val) ? 0 : val;
    } else {
      resultData[i] = resultData[i - axisStride]! + (isNaN(val) ? 0 : val);
    }
  }

  return ArrayStorage.fromData(resultData, [...shape], 'float64');
}

/**
 * Return cumulative product, treating NaNs as one
 * For complex arrays: NaN values (either part NaN) are treated as 1+0i
 */
export function nancumprod(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (isComplexDType(dtype)) {
    // Complex nancumprod - treat NaN values as 1+0i
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumprod
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let prodRe = 1;
      let prodIm = 0;
      for (let i = 0; i < size; i++) {
        const re = complexData[i * 2]!;
        const im = complexData[i * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
          const newRe = prodRe * re - prodIm * im;
          const newIm = prodRe * im + prodIm * re;
          prodRe = newRe;
          prodIm = newIm;
        }
        resultData[i * 2] = prodRe;
        resultData[i * 2 + 1] = prodIm;
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }

    // Perform cumprod along axis
    const totalSize = storage.size;
    const axisStride = strides[normalizedAxis]!;

    for (let i = 0; i < totalSize; i++) {
      const re = complexData[i * 2]!;
      const im = complexData[i * 2 + 1]!;
      const axisPos = Math.floor(i / axisStride) % axisSize;
      const isNan = complexIsNaN(re, im);

      if (axisPos === 0) {
        // If NaN, treat as 1+0i
        resultData[i * 2] = isNan ? 1 : re;
        resultData[i * 2 + 1] = isNan ? 0 : im;
      } else {
        // Multiply by previous element (or 1+0i if NaN)
        const prevRe = resultData[(i - axisStride) * 2]!;
        const prevIm = resultData[(i - axisStride) * 2 + 1]!;
        if (isNan) {
          // Keep previous value (multiply by 1+0i)
          resultData[i * 2] = prevRe;
          resultData[i * 2 + 1] = prevIm;
        } else {
          // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
          resultData[i * 2] = prevRe * re - prevIm * im;
          resultData[i * 2 + 1] = prevRe * im + prevIm * re;
        }
      }
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Flatten and cumprod
    const size = storage.size;
    const resultData = new Float64Array(size);
    let prod = 1;
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        prod *= val;
      }
      resultData[i] = prod;
    }
    return ArrayStorage.fromData(resultData, [size], 'float64');
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape
  const resultData = new Float64Array(storage.size);
  const axisSize = shape[normalizedAxis]!;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Perform cumprod along axis
  const totalSize = storage.size;
  const axisStride = strides[normalizedAxis]!;

  for (let i = 0; i < totalSize; i++) {
    const val = Number(data[i]);
    const axisPos = Math.floor(i / axisStride) % axisSize;

    if (axisPos === 0) {
      resultData[i] = isNaN(val) ? 1 : val;
    } else {
      resultData[i] = resultData[i - axisStride]! * (isNaN(val) ? 1 : val);
    }
  }

  return ArrayStorage.fromData(resultData, [...shape], 'float64');
}

/**
 * Compute median ignoring NaN values
 */
export function nanmedian(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  throwIfComplex(storage.dtype, 'nanmedian', 'Complex numbers are not orderable.');
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;

  if (axis === undefined) {
    // Collect non-NaN values
    const values: number[] = [];
    for (let i = 0; i < storage.size; i++) {
      const val = Number(data[i]);
      if (!isNaN(val)) {
        values.push(val);
      }
    }

    if (values.length === 0) return NaN;

    values.sort((a, b) => a - b);
    const n = values.length;
    const mid = Math.floor(n / 2);

    if (n % 2 === 0) {
      return (values[mid - 1]! + values[mid]!) / 2;
    }
    return values[mid]!;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanmedian(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const resultData = new Float64Array(outerSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    // Collect non-NaN values along axis
    const values: number[] = [];
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      const val = Number(data[linearIdx]);
      if (!isNaN(val)) {
        values.push(val);
      }
    }

    if (values.length === 0) {
      resultData[outerIdx] = NaN;
      continue;
    }

    values.sort((a, b) => a - b);
    const n = values.length;
    const mid = Math.floor(n / 2);

    if (n % 2 === 0) {
      resultData[outerIdx] = (values[mid - 1]! + values[mid]!) / 2;
    } else {
      resultData[outerIdx] = values[mid]!;
    }
  }

  const result = ArrayStorage.fromData(resultData, outputShape, 'float64');

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}
