/**
 * Sorting and searching operations
 *
 * Functions for sorting arrays, finding sorted indices, and searching.
 * @module ops/sorting
 */

import { ArrayStorage } from '../storage';
import { isBigIntDType, isComplexDType, type DType } from '../dtype';
import { Complex } from '../complex';
import {
  outerIndexToMultiIndex,
  multiIndexToLinear,
  multiIndexToBuffer,
} from '../internal/indexing';

/**
 * Check if a value at index i is non-zero (truthy)
 * For complex arrays, checks if either real or imag part is non-zero
 */
function isNonZero(data: ArrayStorage['data'], index: number, isComplex: boolean): boolean {
  if (isComplex) {
    const re = (data as Float64Array | Float32Array)[index * 2]!;
    const im = (data as Float64Array | Float32Array)[index * 2 + 1]!;
    return re !== 0 || im !== 0;
  }
  return Boolean(data[index]);
}

/**
 * Lexicographic comparison for complex numbers
 * Compares real parts first, then imaginary parts
 * NaN values sort to the end
 */
function complexCompare(aRe: number, aIm: number, bRe: number, bIm: number): number {
  const aIsNaN = isNaN(aRe) || isNaN(aIm);
  const bIsNaN = isNaN(bRe) || isNaN(bIm);

  // NaN values go to end
  if (aIsNaN && bIsNaN) return 0;
  if (aIsNaN) return 1;
  if (bIsNaN) return -1;

  // Lexicographic: compare real first, then imaginary
  if (aRe < bRe) return -1;
  if (aRe > bRe) return 1;
  if (aIm < bIm) return -1;
  if (aIm > bIm) return 1;
  return 0;
}

/**
 * Return a sorted copy of an array
 * @param storage - Input array storage
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Sorted array
 */
export function sort(storage: ArrayStorage, axis: number = -1): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Handle 0-d arrays
  if (ndim === 0) {
    return storage.copy();
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create output storage
  const result = storage.copy();
  const resultData = result.data;

  const axisSize = shape[normalizedAxis]!;

  // Compute output shape (same as input)
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

  // Sort along axis
  if (isComplexDType(dtype)) {
    // Complex sort using lexicographic ordering
    const complexData = data as Float64Array | Float32Array;
    const resultComplex = resultData as Float64Array | Float32Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis
      const values: { re: number; im: number; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({
          re: complexData[bufIdx * 2]!,
          im: complexData[bufIdx * 2 + 1]!,
          idx: axisIdx,
        });
      }

      // Sort using lexicographic comparison
      values.sort((a, b) => complexCompare(a.re, a.im, b.re, b.im));

      // Write sorted values back (result is contiguous copy)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultComplex[linearIdx * 2] = values[axisIdx]!.re;
        resultComplex[linearIdx * 2 + 1] = values[axisIdx]!.im;
      }
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis
      const values: { value: bigint; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({ value: typedData[bufIdx]!, idx: axisIdx });
      }

      // Sort by value
      values.sort((a, b) => (a.value < b.value ? -1 : a.value > b.value ? 1 : 0));

      // Write sorted values back (result is contiguous copy)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultTyped[linearIdx] = values[axisIdx]!.value;
      }
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis
      const values: number[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push(Number(data[bufIdx]!));
      }

      // Sort (NaN values go to end)
      values.sort((a, b) => {
        if (isNaN(a) && isNaN(b)) return 0;
        if (isNaN(a)) return 1;
        if (isNaN(b)) return -1;
        return a - b;
      });

      // Write sorted values back (result is contiguous copy)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!;
      }
    }
  }

  return result;
}

/**
 * Returns the indices that would sort an array
 * @param storage - Input array storage
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Array of indices that sort the input array
 */
export function argsort(storage: ArrayStorage, axis: number = -1): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Handle 0-d arrays
  if (ndim === 0) {
    return ArrayStorage.zeros([0], 'int32');
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create output storage with int32 dtype
  const result = ArrayStorage.zeros(Array.from(shape), 'int32');
  const resultData = result.data as Int32Array;

  const axisSize = shape[normalizedAxis]!;

  // Compute outer iteration
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

  // Get argsort along axis
  if (isComplexDType(dtype)) {
    // Complex argsort using lexicographic ordering
    const complexData = data as Float64Array | Float32Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis with their indices
      const values: { re: number; im: number; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({
          re: complexData[bufIdx * 2]!,
          im: complexData[bufIdx * 2 + 1]!,
          idx: axisIdx,
        });
      }

      // Sort using lexicographic comparison
      values.sort((a, b) => complexCompare(a.re, a.im, b.re, b.im));

      // Write sorted indices back (result is contiguous)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!.idx;
      }
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis with their indices
      const values: { value: bigint; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({ value: typedData[bufIdx]!, idx: axisIdx });
      }

      // Sort by value
      values.sort((a, b) => (a.value < b.value ? -1 : a.value > b.value ? 1 : 0));

      // Write sorted indices back (result is contiguous)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!.idx;
      }
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis with their indices
      const values: { value: number; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({ value: Number(data[bufIdx]!), idx: axisIdx });
      }

      // Sort by value (NaN values go to end)
      values.sort((a, b) => {
        if (isNaN(a.value) && isNaN(b.value)) return 0;
        if (isNaN(a.value)) return 1;
        if (isNaN(b.value)) return -1;
        return a.value - b.value;
      });

      // Write sorted indices back (result is contiguous)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!.idx;
      }
    }
  }

  return result;
}

/**
 * Perform an indirect stable sort using a sequence of keys
 * @param keys - Array of ArrayStorage, the last key is the primary sort key
 * @returns Array of indices that would sort the keys
 */
export function lexsort(keys: ArrayStorage[]): ArrayStorage {
  if (keys.length === 0) {
    return ArrayStorage.zeros([0], 'int32');
  }

  // All keys must be 1D with the same length
  const firstKey = keys[0]!;
  const n = firstKey.size;

  for (const key of keys) {
    if (key.ndim !== 1) {
      throw new Error('keys must be 1D arrays');
    }
    if (key.size !== n) {
      throw new Error('all keys must have the same length');
    }
  }

  // Create indices array
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    indices.push(i);
  }

  // Sort using all keys (last key is primary, first key is secondary)
  // Pre-check which keys are complex and extract contiguity info
  const keyIsComplex = keys.map((k) => isComplexDType(k.dtype));
  const keyContiguous = keys.map((k) => k.isCContiguous);
  const keyData = keys.map((k) => k.data);
  const keyOff = keys.map((k) => k.offset);

  indices.sort((a, b) => {
    // Iterate keys in reverse order (last is primary)
    for (let k = keys.length - 1; k >= 0; k--) {
      if (keyIsComplex[k]) {
        let aRe: number, aIm: number, bRe: number, bIm: number;
        if (keyContiguous[k]) {
          const d = keyData[k]! as Float64Array | Float32Array;
          const o = keyOff[k]!;
          aRe = d[(o + a) * 2]!;
          aIm = d[(o + a) * 2 + 1]!;
          bRe = d[(o + b) * 2]!;
          bIm = d[(o + b) * 2 + 1]!;
        } else {
          const va = keys[k]!.iget(a) as Complex;
          const vb = keys[k]!.iget(b) as Complex;
          aRe = va.re;
          aIm = va.im;
          bRe = vb.re;
          bIm = vb.im;
        }
        if (aRe < bRe) return -1;
        if (aRe > bRe) return 1;
        if (aIm < bIm) return -1;
        if (aIm > bIm) return 1;
      } else {
        let va: number, vb: number;
        if (keyContiguous[k]) {
          const d = keyData[k]!;
          const o = keyOff[k]!;
          va = Number(d[o + a]!);
          vb = Number(d[o + b]!);
        } else {
          va = Number(keys[k]!.iget(a));
          vb = Number(keys[k]!.iget(b));
        }

        // Handle NaN (put at end)
        if (isNaN(va) && isNaN(vb)) continue;
        if (isNaN(va)) return 1;
        if (isNaN(vb)) return -1;

        if (va < vb) return -1;
        if (va > vb) return 1;
      }
      // Equal, continue to next key
    }
    return 0; // Stable sort - preserve original order
  });

  // Create result
  const result = ArrayStorage.zeros([n], 'int32');
  const resultData = result.data as Int32Array;
  for (let i = 0; i < n; i++) {
    resultData[i] = indices[i]!;
  }

  return result;
}

/**
 * Quickselect algorithm helper for number arrays
 * Partitions array so element at kth position is in sorted position
 */
function quickselectNumbers(arr: number[], kth: number): void {
  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    // Choose pivot using median-of-three
    const mid = Math.floor((left + right) / 2);
    const a = arr[left]!;
    const b = arr[mid]!;
    const c = arr[right]!;

    // Sort three elements and use middle as pivot
    let pivotIdx: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotIdx = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotIdx = left;
    else pivotIdx = right;

    // Move pivot to end
    const pivot = arr[pivotIdx]!;
    [arr[pivotIdx], arr[right]] = [arr[right]!, arr[pivotIdx]!];

    // Partition: all elements <= pivot go to left
    let i = left;
    for (let j = left; j < right; j++) {
      const val = arr[j]!;
      // Handle NaN: NaN values go to the end
      const valIsNaN = isNaN(val);
      const pivotIsNaN = isNaN(pivot);

      if (!valIsNaN && (pivotIsNaN || val <= pivot)) {
        [arr[i], arr[j]] = [arr[j]!, arr[i]!];
        i++;
      }
    }
    [arr[i], arr[right]] = [arr[right]!, arr[i]!];

    // Recurse on appropriate side
    if (i === kth) {
      return;
    } else if (i < kth) {
      left = i + 1;
    } else {
      right = i - 1;
    }
  }
}

/**
 * Quickselect algorithm helper for bigint arrays
 * Partitions array so element at kth position is in sorted position
 */
function quickselectBigInts(arr: bigint[], kth: number): void {
  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    // Choose pivot using median-of-three
    const mid = Math.floor((left + right) / 2);
    const a = arr[left]!;
    const b = arr[mid]!;
    const c = arr[right]!;

    // Sort three elements and use middle as pivot
    let pivotIdx: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotIdx = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotIdx = left;
    else pivotIdx = right;

    // Move pivot to end
    const pivot = arr[pivotIdx]!;
    [arr[pivotIdx], arr[right]] = [arr[right]!, arr[pivotIdx]!];

    // Partition: all elements <= pivot go to left
    let i = left;
    for (let j = left; j < right; j++) {
      if (arr[j]! <= pivot) {
        [arr[i], arr[j]] = [arr[j]!, arr[i]!];
        i++;
      }
    }
    [arr[i], arr[right]] = [arr[right]!, arr[i]!];

    // Recurse on appropriate side
    if (i === kth) {
      return;
    } else if (i < kth) {
      left = i + 1;
    } else {
      right = i - 1;
    }
  }
}

/**
 * Quickselect for argpartition with number values
 * Partitions array of {value, idx} pairs by value
 */
function quickselectNumberIndices(arr: { value: number; idx: number }[], kth: number): void {
  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    // Choose pivot using median-of-three
    const mid = Math.floor((left + right) / 2);
    const a = arr[left]!.value;
    const b = arr[mid]!.value;
    const c = arr[right]!.value;

    // Sort three elements and use middle as pivot
    let pivotIdx: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotIdx = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotIdx = left;
    else pivotIdx = right;

    // Move pivot to end
    const pivot = arr[pivotIdx]!.value;
    [arr[pivotIdx], arr[right]] = [arr[right]!, arr[pivotIdx]!];

    // Partition: all elements <= pivot go to left
    let i = left;
    for (let j = left; j < right; j++) {
      const val = arr[j]!.value;
      // Handle NaN: NaN values go to the end
      const valIsNaN = isNaN(val);
      const pivotIsNaN = isNaN(pivot);

      if (!valIsNaN && (pivotIsNaN || val <= pivot)) {
        [arr[i], arr[j]] = [arr[j]!, arr[i]!];
        i++;
      }
    }
    [arr[i], arr[right]] = [arr[right]!, arr[i]!];

    // Recurse on appropriate side
    if (i === kth) {
      return;
    } else if (i < kth) {
      left = i + 1;
    } else {
      right = i - 1;
    }
  }
}

/**
 * Quickselect for argpartition with bigint values
 * Partitions array of {value, idx} pairs by value
 */
function quickselectBigIntIndices(arr: { value: bigint; idx: number }[], kth: number): void {
  let left = 0;
  let right = arr.length - 1;

  while (left < right) {
    // Choose pivot using median-of-three
    const mid = Math.floor((left + right) / 2);
    const a = arr[left]!.value;
    const b = arr[mid]!.value;
    const c = arr[right]!.value;

    // Sort three elements and use middle as pivot
    let pivotIdx: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotIdx = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotIdx = left;
    else pivotIdx = right;

    // Move pivot to end
    const pivot = arr[pivotIdx]!.value;
    [arr[pivotIdx], arr[right]] = [arr[right]!, arr[pivotIdx]!];

    // Partition: all elements <= pivot go to left
    let i = left;
    for (let j = left; j < right; j++) {
      if (arr[j]!.value <= pivot) {
        [arr[i], arr[j]] = [arr[j]!, arr[i]!];
        i++;
      }
    }
    [arr[i], arr[right]] = [arr[right]!, arr[i]!];

    // Recurse on appropriate side
    if (i === kth) {
      return;
    } else if (i < kth) {
      left = i + 1;
    } else {
      right = i - 1;
    }
  }
}

/**
 * Partially sort an array
 * Returns array with element at kth position in sorted position,
 * all smaller elements before it, all larger after it (not fully sorted)
 * @param storage - Input array storage
 * @param kth - Element index to partition by
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Partitioned array
 */
export function partition(storage: ArrayStorage, kth: number, axis: number = -1): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  // Handle 0-d arrays
  if (ndim === 0) {
    return storage.copy();
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  // Normalize kth
  let normalizedKth = kth;
  if (normalizedKth < 0) {
    normalizedKth = axisSize + normalizedKth;
  }
  if (normalizedKth < 0 || normalizedKth >= axisSize) {
    throw new Error(`kth(=${kth}) out of bounds (${axisSize})`);
  }

  // Create output storage
  const result = storage.copy();
  const resultData = result.data;

  // Compute outer iteration
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

  // Partition along axis using quickselect
  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis
      const values: bigint[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push(resultTyped[linearIdx]!);
      }

      // Partition using quickselect
      quickselectBigInts(values, normalizedKth);

      // Write partitioned values back
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultTyped[linearIdx] = values[axisIdx]!;
      }
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis
      const values: number[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push(Number(resultData[linearIdx]!));
      }

      // Partition using quickselect
      quickselectNumbers(values, normalizedKth);

      // Write partitioned values back
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!;
      }
    }
  }

  return result;
}

/**
 * Returns indices that would partition an array
 * @param storage - Input array storage
 * @param kth - Element index to partition by
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Array of indices
 */
export function argpartition(storage: ArrayStorage, kth: number, axis: number = -1): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Handle 0-d arrays
  if (ndim === 0) {
    return ArrayStorage.zeros([0], 'int32');
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  // Normalize kth
  let normalizedKth = kth;
  if (normalizedKth < 0) {
    normalizedKth = axisSize + normalizedKth;
  }
  if (normalizedKth < 0 || normalizedKth >= axisSize) {
    throw new Error(`kth(=${kth}) out of bounds (${axisSize})`);
  }

  // Create output storage with int32 dtype
  const result = ArrayStorage.zeros(Array.from(shape), 'int32');
  const resultData = result.data as Int32Array;

  // Compute outer iteration
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

  // Get argpartition along axis
  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis with their indices
      const values: { value: bigint; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({ value: typedData[bufIdx]!, idx: axisIdx });
      }

      // Partition using quickselect
      quickselectBigIntIndices(values, normalizedKth);

      // Write partitioned indices back (result is contiguous)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!.idx;
      }
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis with their indices
      const values: { value: number; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
        values.push({ value: Number(data[bufIdx]!), idx: axisIdx });
      }

      // Partition using quickselect
      quickselectNumberIndices(values, normalizedKth);

      // Write partitioned indices back (result is contiguous)
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        resultData[linearIdx] = values[axisIdx]!.idx;
      }
    }
  }

  return result;
}

/**
 * Sort a complex array using the real part first, then the imaginary part
 * For real arrays, this is equivalent to sort
 * @param storage - Input array storage
 * @returns Sorted array
 */
export function sort_complex(storage: ArrayStorage): ArrayStorage {
  const dtype = storage.dtype;
  const size = storage.size;
  const contiguous = storage.isCContiguous;
  const data = storage.data;
  const off = storage.offset;

  if (isComplexDType(dtype)) {
    const values: { re: number; im: number }[] = [];
    if (contiguous) {
      const typedData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        values.push({ re: typedData[(off + i) * 2]!, im: typedData[(off + i) * 2 + 1]! });
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as { re: number; im: number };
        values.push({ re: val.re, im: val.im });
      }
    }

    // Sort using lexicographic comparison
    values.sort((a, b) => complexCompare(a.re, a.im, b.re, b.im));

    // Create result (1D sorted array with complex128 dtype)
    const result = ArrayStorage.zeros([size], 'complex128');
    const resultData = result.data as Float64Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = values[i]!.re;
      resultData[i * 2 + 1] = values[i]!.im;
    }

    return result;
  } else {
    // For real arrays, sort normally (1D flattened), then cast to complex128
    const values: number[] = [];
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        values.push(Number(data[off + i]!));
      }
    } else {
      for (let i = 0; i < size; i++) {
        values.push(Number(storage.iget(i)));
      }
    }

    // Sort (NaN values go to end)
    values.sort((a, b) => {
      if (isNaN(a) && isNaN(b)) return 0;
      if (isNaN(a)) return 1;
      if (isNaN(b)) return -1;
      return a - b;
    });

    // Create result as complex128 (NumPy always returns complex)
    const result = ArrayStorage.zeros([size], 'complex128');
    const resultData = result.data as Float64Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = values[i]!;
      resultData[i * 2 + 1] = 0;
    }

    return result;
  }
}

// ============================================================================
// Searching operations
// ============================================================================

/**
 * Return the indices of the elements that are non-zero
 * @param storage - Input array storage
 * @returns Tuple of arrays, one for each dimension
 */
export function nonzero(storage: ArrayStorage): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const isComplex = isComplexDType(storage.dtype);
  const contiguous = storage.isCContiguous;
  const data = storage.data;
  const off = storage.offset;

  // Find all non-zero indices
  const nonzeroIndices: number[][] = [];
  for (let dim = 0; dim < ndim; dim++) {
    nonzeroIndices.push([]);
  }

  // Calculate strides for logical index conversion
  const logicalStrides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    logicalStrides.unshift(stride);
    stride *= shape[i]!;
  }

  // Find non-zero elements
  if (contiguous) {
    for (let i = 0; i < size; i++) {
      if (isNonZero(data, off + i, isComplex)) {
        let remaining = i;
        for (let dim = 0; dim < ndim; dim++) {
          const idx = Math.floor(remaining / logicalStrides[dim]!);
          remaining = remaining % logicalStrides[dim]!;
          nonzeroIndices[dim]!.push(idx);
        }
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = storage.iget(i);
      const nz = isComplex
        ? (val as { re: number; im: number }).re !== 0 ||
          (val as { re: number; im: number }).im !== 0
        : Boolean(val);
      if (nz) {
        let remaining = i;
        for (let dim = 0; dim < ndim; dim++) {
          const idx = Math.floor(remaining / logicalStrides[dim]!);
          remaining = remaining % logicalStrides[dim]!;
          nonzeroIndices[dim]!.push(idx);
        }
      }
    }
  }

  // Create result arrays
  const numNonzero = nonzeroIndices[0]?.length ?? 0;
  const result: ArrayStorage[] = [];

  for (let dim = 0; dim < ndim; dim++) {
    const arr = ArrayStorage.zeros([numNonzero], 'int32');
    const arrData = arr.data as Int32Array;
    for (let i = 0; i < numNonzero; i++) {
      arrData[i] = nonzeroIndices[dim]![i]!;
    }
    result.push(arr);
  }

  return result;
}

/**
 * Find the indices of array elements that are non-zero, grouped by element
 * Returns a 2D array where each row is the index of a non-zero element.
 * This is equivalent to transpose(nonzero(a)).
 * @param storage - Input array storage
 * @returns 2D array of shape (N, ndim) where N is number of non-zero elements
 */
export function argwhere(storage: ArrayStorage): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const isComplex = isComplexDType(storage.dtype);
  const contiguous = storage.isCContiguous;
  const data = storage.data;
  const off = storage.offset;

  // Find all non-zero indices
  const nonzeroIndices: number[][] = [];

  // Calculate strides for logical index conversion
  const logicalStrides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    logicalStrides.unshift(stride);
    stride *= shape[i]!;
  }

  // Find non-zero elements
  if (contiguous) {
    for (let i = 0; i < size; i++) {
      if (isNonZero(data, off + i, isComplex)) {
        const indices: number[] = [];
        let remaining = i;
        for (let dim = 0; dim < ndim; dim++) {
          const idx = Math.floor(remaining / logicalStrides[dim]!);
          remaining = remaining % logicalStrides[dim]!;
          indices.push(idx);
        }
        nonzeroIndices.push(indices);
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = storage.iget(i);
      const nz = isComplex
        ? (val as { re: number; im: number }).re !== 0 ||
          (val as { re: number; im: number }).im !== 0
        : Boolean(val);
      if (nz) {
        const indices: number[] = [];
        let remaining = i;
        for (let dim = 0; dim < ndim; dim++) {
          const idx = Math.floor(remaining / logicalStrides[dim]!);
          remaining = remaining % logicalStrides[dim]!;
          indices.push(idx);
        }
        nonzeroIndices.push(indices);
      }
    }
  }

  // Create result array: shape is (numNonzero, ndim)
  const numNonzero = nonzeroIndices.length;
  const resultShape = ndim === 0 ? [numNonzero, 1] : [numNonzero, ndim];
  const result = ArrayStorage.zeros(resultShape, 'int32');
  const resultData = result.data as Int32Array;

  // Fill result array
  for (let i = 0; i < numNonzero; i++) {
    const indices = nonzeroIndices[i]!;
    for (let dim = 0; dim < (ndim === 0 ? 1 : ndim); dim++) {
      resultData[i * (ndim === 0 ? 1 : ndim) + dim] = indices[dim] ?? 0;
    }
  }

  return result;
}

/**
 * Return indices of non-zero elements in flattened array
 * @param storage - Input array storage
 * @returns Array of indices
 */
export function flatnonzero(storage: ArrayStorage): ArrayStorage {
  const size = storage.size;
  const isComplex = isComplexDType(storage.dtype);
  const contiguous = storage.isCContiguous;
  const data = storage.data;
  const off = storage.offset;

  // Find all non-zero indices
  const indices: number[] = [];
  if (contiguous) {
    for (let i = 0; i < size; i++) {
      if (isNonZero(data, off + i, isComplex)) {
        indices.push(i);
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = storage.iget(i);
      const nz = isComplex
        ? (val as { re: number; im: number }).re !== 0 ||
          (val as { re: number; im: number }).im !== 0
        : Boolean(val);
      if (nz) {
        indices.push(i);
      }
    }
  }

  // Create result
  const result = ArrayStorage.zeros([indices.length], 'int32');
  const resultData = result.data as Int32Array;
  for (let i = 0; i < indices.length; i++) {
    resultData[i] = indices[i]!;
  }

  return result;
}

/**
 * Return elements from x or y depending on condition
 * @param condition - Boolean array or condition
 * @param x - Values where condition is true
 * @param y - Values where condition is false
 * @returns Array with elements chosen from x or y based on condition
 */
export function where(
  condition: ArrayStorage,
  x?: ArrayStorage,
  y?: ArrayStorage
): ArrayStorage | ArrayStorage[] {
  // If only condition is given, return indices of true elements (like nonzero)
  if (x === undefined && y === undefined) {
    return nonzero(condition);
  }

  // Both x and y must be provided
  if (x === undefined || y === undefined) {
    throw new Error('either both or neither of x and y should be given');
  }

  const condShape = condition.shape;
  const xShape = x.shape;
  const yShape = y.shape;

  // For now, require same shapes (broadcasting could be added later)
  // Find maximum shape for broadcasting
  const maxNdim = Math.max(condShape.length, xShape.length, yShape.length);

  // Pad shapes with 1s at the front for broadcasting
  const padShape = (s: readonly number[]) => {
    const padded = Array(maxNdim).fill(1);
    for (let i = 0; i < s.length; i++) {
      padded[maxNdim - s.length + i] = s[i];
    }
    return padded;
  };

  const paddedCond = padShape(condShape);
  const paddedX = padShape(xShape);
  const paddedY = padShape(yShape);

  // Compute broadcast shape
  const resultShape: number[] = [];
  for (let i = 0; i < maxNdim; i++) {
    const dims = [paddedCond[i]!, paddedX[i]!, paddedY[i]!];
    const maxDim = Math.max(...dims);
    for (const d of dims) {
      if (d !== 1 && d !== maxDim) {
        throw new Error(`operands could not be broadcast together`);
      }
    }
    resultShape.push(maxDim);
  }

  // Use float64 for result dtype (could be improved to handle dtype promotion)
  const resultDtype = x.dtype as DType;
  const result = ArrayStorage.zeros(resultShape, resultDtype);
  const resultData = result.data;

  // Calculate C-contiguous logical strides for broadcasting index computation
  const calcLogicalStrides = (shape: readonly number[], padded: number[]) => {
    const strides: number[] = [];
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }
    // Pad with zeros for dimensions that don't exist
    while (strides.length < padded.length) {
      strides.unshift(0);
    }
    // Set stride to 0 for broadcast dimensions
    for (let i = 0; i < padded.length; i++) {
      if (padded[i] === 1 && resultShape[i] !== 1) {
        strides[i] = 0;
      }
    }
    return strides;
  };

  const condLogicalStrides = calcLogicalStrides(condShape, paddedCond);
  const xLogicalStrides = calcLogicalStrides(xShape, paddedX);
  const yLogicalStrides = calcLogicalStrides(yShape, paddedY);

  // Calculate result strides
  const resultStrides: number[] = [];
  let stride = 1;
  for (let i = resultShape.length - 1; i >= 0; i--) {
    resultStrides.unshift(stride);
    stride *= resultShape[i]!;
  }

  const totalSize = resultShape.reduce((a, b) => a * b, 1);
  const isCondComplex = isComplexDType(condition.dtype);
  const isResultComplex = isComplexDType(resultDtype);

  // Pre-extract contiguity info for fast paths
  const condContiguous = condition.isCContiguous;
  const condData = condition.data;
  const condOff = condition.offset;

  const xContiguous = x.isCContiguous;
  const xData = x.data;
  const xOff = x.offset;

  const yContiguous = y.isCContiguous;
  const yData = y.data;
  const yOff = y.offset;

  // Check if all three inputs are contiguous and shapes match (no broadcasting needed)
  const noBroadcast =
    condShape.length === resultShape.length &&
    xShape.length === resultShape.length &&
    yShape.length === resultShape.length &&
    condShape.every((d, i) => d === resultShape[i]) &&
    xShape.every((d, i) => d === resultShape[i]) &&
    yShape.every((d, i) => d === resultShape[i]);

  if (noBroadcast && condContiguous && xContiguous && yContiguous) {
    // Fast path: all inputs are contiguous and same shape
    if (isResultComplex) {
      const resultTyped = resultData as Float64Array | Float32Array;
      const xTyped = xData as Float64Array | Float32Array;
      const yTyped = yData as Float64Array | Float32Array;
      for (let i = 0; i < totalSize; i++) {
        const condNz = isNonZero(condData, condOff + i, isCondComplex);
        if (condNz) {
          resultTyped[i * 2] = xTyped[(xOff + i) * 2]!;
          resultTyped[i * 2 + 1] = xTyped[(xOff + i) * 2 + 1]!;
        } else {
          resultTyped[i * 2] = yTyped[(yOff + i) * 2]!;
          resultTyped[i * 2 + 1] = yTyped[(yOff + i) * 2 + 1]!;
        }
      }
    } else if (isBigIntDType(resultDtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      const xTyped = xData as BigInt64Array | BigUint64Array;
      const yTyped = yData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < totalSize; i++) {
        const condNz = isNonZero(condData, condOff + i, isCondComplex);
        if (condNz) {
          resultTyped[i] = xTyped[xOff + i]!;
        } else {
          resultTyped[i] = yTyped[yOff + i]!;
        }
      }
    } else {
      for (let i = 0; i < totalSize; i++) {
        const condNz = isNonZero(condData, condOff + i, isCondComplex);
        if (condNz) {
          resultData[i] = xData[xOff + i] as number;
        } else {
          resultData[i] = yData[yOff + i] as number;
        }
      }
    }
  } else {
    // General path with broadcasting — use per-array contiguity checks
    for (let i = 0; i < totalSize; i++) {
      let remaining = i;
      let condIdx = 0;
      let xIdx = 0;
      let yIdx = 0;

      for (let dim = 0; dim < maxNdim; dim++) {
        const idx = Math.floor(remaining / resultStrides[dim]!);
        remaining = remaining % resultStrides[dim]!;

        condIdx += idx * condLogicalStrides[dim]!;
        xIdx += idx * xLogicalStrides[dim]!;
        yIdx += idx * yLogicalStrides[dim]!;
      }

      // Check condition
      let condTruthy: boolean;
      if (condContiguous) {
        condTruthy = isNonZero(condData, condOff + condIdx, isCondComplex);
      } else {
        const condVal = condition.iget(condIdx);
        condTruthy = isCondComplex
          ? (condVal as { re: number; im: number }).re !== 0 ||
            (condVal as { re: number; im: number }).im !== 0
          : Boolean(condVal);
      }

      if (condTruthy) {
        if (isResultComplex) {
          if (xContiguous) {
            const xTyped = xData as Float64Array | Float32Array;
            (resultData as Float64Array | Float32Array)[i * 2] = xTyped[(xOff + xIdx) * 2]!;
            (resultData as Float64Array | Float32Array)[i * 2 + 1] = xTyped[(xOff + xIdx) * 2 + 1]!;
          } else {
            const val = x.iget(xIdx) as { re: number; im: number };
            (resultData as Float64Array | Float32Array)[i * 2] = val.re;
            (resultData as Float64Array | Float32Array)[i * 2 + 1] = val.im;
          }
        } else {
          if (xContiguous) {
            resultData[i] = xData[xOff + xIdx] as number;
          } else {
            resultData[i] = x.iget(xIdx) as number;
          }
        }
      } else {
        if (isResultComplex) {
          if (yContiguous) {
            const yTyped = yData as Float64Array | Float32Array;
            (resultData as Float64Array | Float32Array)[i * 2] = yTyped[(yOff + yIdx) * 2]!;
            (resultData as Float64Array | Float32Array)[i * 2 + 1] = yTyped[(yOff + yIdx) * 2 + 1]!;
          } else {
            const val = y.iget(yIdx) as { re: number; im: number };
            (resultData as Float64Array | Float32Array)[i * 2] = val.re;
            (resultData as Float64Array | Float32Array)[i * 2 + 1] = val.im;
          }
        } else {
          if (yContiguous) {
            resultData[i] = yData[yOff + yIdx] as number;
          } else {
            resultData[i] = y.iget(yIdx) as number;
          }
        }
      }
    }
  }

  return result;
}

/**
 * Find indices where elements should be inserted to maintain order
 * @param storage - Input array (must be sorted in ascending order)
 * @param values - Values to insert
 * @param side - 'left' or 'right' side to insert
 * @returns Indices where values should be inserted
 */
export function searchsorted(
  storage: ArrayStorage,
  values: ArrayStorage,
  side: 'left' | 'right' = 'left'
): ArrayStorage {
  // Input array must be 1D
  if (storage.ndim !== 1) {
    throw new Error('storage must be 1D');
  }

  const n = storage.size;
  const numValues = values.size;
  const isComplex = isComplexDType(storage.dtype);

  // Pre-extract contiguity info
  const storageContiguous = storage.isCContiguous;
  const storageData = storage.data;
  const storageOff = storage.offset;

  const valuesContiguous = values.isCContiguous;
  const valuesData = values.data;
  const valuesOff = values.offset;

  // Create result array
  const result = ArrayStorage.zeros([numValues], 'int32');
  const resultData = result.data as Int32Array;

  if (isComplex) {
    if (storageContiguous && valuesContiguous) {
      // Fast path: both arrays are contiguous
      const sData = storageData as Float64Array | Float32Array;
      const vData = valuesData as Float64Array | Float32Array;

      for (let i = 0; i < numValues; i++) {
        const vRe = vData[(valuesOff + i) * 2]!;
        const vIm = vData[(valuesOff + i) * 2 + 1]!;
        let lo = 0;
        let hi = n;

        if (side === 'left') {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            const midRe = sData[(storageOff + mid) * 2]!;
            const midIm = sData[(storageOff + mid) * 2 + 1]!;
            if (complexCompare(midRe, midIm, vRe, vIm) < 0) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        } else {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            const midRe = sData[(storageOff + mid) * 2]!;
            const midIm = sData[(storageOff + mid) * 2 + 1]!;
            if (complexCompare(midRe, midIm, vRe, vIm) <= 0) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        }

        resultData[i] = lo;
      }
    } else {
      // Slow path: use iget for non-contiguous views
      for (let i = 0; i < numValues; i++) {
        const v = values.iget(i) as { re: number; im: number };
        const vRe = v.re;
        const vIm = v.im;
        let lo = 0;
        let hi = n;

        if (side === 'left') {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            const midVal = storage.iget(mid) as { re: number; im: number };
            if (complexCompare(midVal.re, midVal.im, vRe, vIm) < 0) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        } else {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            const midVal = storage.iget(mid) as { re: number; im: number };
            if (complexCompare(midVal.re, midVal.im, vRe, vIm) <= 0) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        }

        resultData[i] = lo;
      }
    }
  } else {
    if (storageContiguous && valuesContiguous) {
      // Fast path: both arrays are contiguous
      for (let i = 0; i < numValues; i++) {
        const v = Number(valuesData[valuesOff + i]!);
        let lo = 0;
        let hi = n;

        if (side === 'left') {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (Number(storageData[storageOff + mid]!) < v) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        } else {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (Number(storageData[storageOff + mid]!) <= v) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        }

        resultData[i] = lo;
      }
    } else {
      // Slow path: use iget for non-contiguous views
      for (let i = 0; i < numValues; i++) {
        const v = Number(values.iget(i));
        let lo = 0;
        let hi = n;

        if (side === 'left') {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (Number(storage.iget(mid)) < v) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        } else {
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (Number(storage.iget(mid)) <= v) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
        }

        resultData[i] = lo;
      }
    }
  }

  return result;
}

/**
 * Return the elements of an array that satisfy some condition
 * @param condition - Boolean array
 * @param storage - Input array storage
 * @returns 1D array of elements where condition is true
 */
export function extract(condition: ArrayStorage, storage: ArrayStorage): ArrayStorage {
  const dtype = storage.dtype;
  const isCondComplex = isComplexDType(condition.dtype);
  const isDataComplex = isComplexDType(dtype);

  // Both arrays should have same size
  const size = Math.min(condition.size, storage.size);

  // Pre-extract contiguity info
  const condContiguous = condition.isCContiguous;
  const condData = condition.data;
  const condOff = condition.offset;

  const dataContiguous = storage.isCContiguous;
  const data = storage.data;
  const dataOff = storage.offset;

  // Count number of true values
  let count = 0;
  if (condContiguous) {
    for (let i = 0; i < size; i++) {
      if (isNonZero(condData, condOff + i, isCondComplex)) count++;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = condition.iget(i);
      const nz = isCondComplex
        ? (val as { re: number; im: number }).re !== 0 ||
          (val as { re: number; im: number }).im !== 0
        : Boolean(val);
      if (nz) count++;
    }
  }

  // Create result array
  const result = ArrayStorage.zeros([count], dtype as DType);
  const resultData = result.data;

  // Copy values where condition is true
  let idx = 0;
  if (condContiguous && dataContiguous) {
    // Fast path: both condition and data are contiguous
    if (isBigIntDType(dtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      const typedData = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        if (isNonZero(condData, condOff + i, isCondComplex)) {
          resultTyped[idx++] = typedData[dataOff + i]!;
        }
      }
    } else if (isDataComplex) {
      const resultTyped = resultData as Float64Array | Float32Array;
      const typedData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        if (isNonZero(condData, condOff + i, isCondComplex)) {
          resultTyped[idx * 2] = typedData[(dataOff + i) * 2]!;
          resultTyped[idx * 2 + 1] = typedData[(dataOff + i) * 2 + 1]!;
          idx++;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        if (isNonZero(condData, condOff + i, isCondComplex)) {
          resultData[idx++] = data[dataOff + i] as number;
        }
      }
    }
  } else {
    // Slow path: use iget for non-contiguous views
    const condTruthy = condContiguous
      ? (i: number): boolean => isNonZero(condData, condOff + i, isCondComplex)
      : (i: number): boolean => {
          const val = condition.iget(i);
          if (isCondComplex) {
            const c = val as { re: number; im: number };
            return c.re !== 0 || c.im !== 0;
          }
          return Boolean(val);
        };

    if (isBigIntDType(dtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        if (condTruthy(i)) {
          resultTyped[idx++] = storage.iget(i) as bigint;
        }
      }
    } else if (isDataComplex) {
      const resultTyped = resultData as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        if (condTruthy(i)) {
          const val = storage.iget(i) as { re: number; im: number };
          resultTyped[idx * 2] = val.re;
          resultTyped[idx * 2 + 1] = val.im;
          idx++;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        if (condTruthy(i)) {
          resultData[idx++] = storage.iget(i) as number;
        }
      }
    }
  }

  return result;
}

/**
 * Count number of non-zero values in the array
 * @param storage - Input array storage
 * @param axis - Axis along which to count (optional)
 * @returns Count of non-zero values
 */
export function count_nonzero(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const size = storage.size;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const isComplex = isComplexDType(storage.dtype);
  const contiguous = storage.isCContiguous;

  if (axis === undefined) {
    // Count all non-zero elements
    let count = 0;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        if (isNonZero(data, off + i, isComplex)) {
          count++;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i);
        if (isComplex) {
          const c = val as { re: number; im: number };
          if (c.re !== 0 || c.im !== 0) count++;
        } else {
          if (val !== 0 && val !== BigInt(0)) count++;
        }
      }
    }
    return count;
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
    return count_nonzero(storage);
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data as Int32Array;

  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let count = 0;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
      const bufIdx = multiIndexToBuffer(inputIndices, inputStrides, off);
      if (isNonZero(data, bufIdx, isComplex)) {
        count++;
      }
    }
    resultData[outerIdx] = count;
  }

  return result;
}
