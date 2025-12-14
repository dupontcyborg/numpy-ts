/**
 * Sorting and searching operations
 *
 * Functions for sorting arrays, finding sorted indices, and searching.
 * @module ops/sorting
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType, isComplexDType, type DType } from '../core/dtype';
import { outerIndexToMultiIndex, multiIndexToLinear } from '../internal/indexing';

/**
 * Check if a value at index i is non-zero (truthy)
 * For complex arrays, checks if either real or imag part is non-zero
 */
function isNonZero(
  data: ArrayStorage['data'],
  index: number,
  isComplex: boolean
): boolean {
  if (isComplex) {
    const re = (data as Float64Array | Float32Array)[index * 2]!;
    const im = (data as Float64Array | Float32Array)[index * 2 + 1]!;
    return re !== 0 || im !== 0;
  }
  return Boolean(data[index]);
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
  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis
      const values: { value: bigint; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push({ value: typedData[linearIdx]!, idx: axisIdx });
      }

      // Sort by value
      values.sort((a, b) => (a.value < b.value ? -1 : a.value > b.value ? 1 : 0));

      // Write sorted values back
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
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push(Number(data[linearIdx]!));
      }

      // Sort (NaN values go to end)
      values.sort((a, b) => {
        if (isNaN(a) && isNaN(b)) return 0;
        if (isNaN(a)) return 1;
        if (isNaN(b)) return -1;
        return a - b;
      });

      // Write sorted values back
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
  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Collect values along axis with their indices
      const values: { value: bigint; idx: number }[] = [];
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const inputIndices = outerIndexToMultiIndex(outerIdx, normalizedAxis, axisIdx, shape);
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push({ value: typedData[linearIdx]!, idx: axisIdx });
      }

      // Sort by value
      values.sort((a, b) => (a.value < b.value ? -1 : a.value > b.value ? 1 : 0));

      // Write sorted indices back
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
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push({ value: Number(data[linearIdx]!), idx: axisIdx });
      }

      // Sort by value (NaN values go to end)
      values.sort((a, b) => {
        if (isNaN(a.value) && isNaN(b.value)) return 0;
        if (isNaN(a.value)) return 1;
        if (isNaN(b.value)) return -1;
        return a.value - b.value;
      });

      // Write sorted indices back
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
  indices.sort((a, b) => {
    // Iterate keys in reverse order (last is primary)
    for (let k = keys.length - 1; k >= 0; k--) {
      const key = keys[k]!;
      const data = key.data;
      const va = Number(data[a]);
      const vb = Number(data[b]);

      // Handle NaN (put at end)
      if (isNaN(va) && isNaN(vb)) continue;
      if (isNaN(va)) return 1;
      if (isNaN(vb)) return -1;

      if (va < vb) return -1;
      if (va > vb) return 1;
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
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push({ value: typedData[linearIdx]!, idx: axisIdx });
      }

      // Partition using quickselect
      quickselectBigIntIndices(values, normalizedKth);

      // Write partitioned indices back
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
        const linearIdx = multiIndexToLinear(inputIndices, shape);
        values.push({ value: Number(data[linearIdx]!), idx: axisIdx });
      }

      // Partition using quickselect
      quickselectNumberIndices(values, normalizedKth);

      // Write partitioned indices back
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
  // For real arrays, just sort normally (1D flattened)
  const dtype = storage.dtype;
  const size = storage.size;
  const data = storage.data;

  // Flatten and sort
  const values: number[] = [];
  for (let i = 0; i < size; i++) {
    values.push(Number(data[i]!));
  }

  // Sort (NaN values go to end)
  values.sort((a, b) => {
    if (isNaN(a) && isNaN(b)) return 0;
    if (isNaN(a)) return 1;
    if (isNaN(b)) return -1;
    return a - b;
  });

  // Create result (1D sorted array)
  const result = ArrayStorage.zeros([size], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < size; i++) {
    resultData[i] = values[i]!;
  }

  return result;
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
  const data = storage.data;
  const size = storage.size;
  const isComplex = isComplexDType(storage.dtype);

  // Find all non-zero indices
  const nonzeroIndices: number[][] = [];
  for (let dim = 0; dim < ndim; dim++) {
    nonzeroIndices.push([]);
  }

  // Calculate strides for index conversion
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Find non-zero elements
  for (let i = 0; i < size; i++) {
    if (isNonZero(data, i, isComplex)) {
      // Convert linear index to multi-index
      let remaining = i;
      for (let dim = 0; dim < ndim; dim++) {
        const idx = Math.floor(remaining / strides[dim]!);
        remaining = remaining % strides[dim]!;
        nonzeroIndices[dim]!.push(idx);
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
  const data = storage.data;
  const size = storage.size;
  const isComplex = isComplexDType(storage.dtype);

  // Find all non-zero indices
  const nonzeroIndices: number[][] = [];

  // Calculate strides for index conversion
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Find non-zero elements
  for (let i = 0; i < size; i++) {
    if (isNonZero(data, i, isComplex)) {
      // Convert linear index to multi-index
      const indices: number[] = [];
      let remaining = i;
      for (let dim = 0; dim < ndim; dim++) {
        const idx = Math.floor(remaining / strides[dim]!);
        remaining = remaining % strides[dim]!;
        indices.push(idx);
      }
      nonzeroIndices.push(indices);
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
  const data = storage.data;
  const size = storage.size;
  const isComplex = isComplexDType(storage.dtype);

  // Find all non-zero indices
  const indices: number[] = [];
  for (let i = 0; i < size; i++) {
    if (isNonZero(data, i, isComplex)) {
      indices.push(i);
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
  const condData = condition.data;
  const xData = x.data;
  const yData = y.data;

  // Calculate strides for broadcasting
  const calcStrides = (shape: readonly number[], padded: number[]) => {
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

  const condStrides = calcStrides(condShape, paddedCond);
  const xStrides = calcStrides(xShape, paddedX);
  const yStrides = calcStrides(yShape, paddedY);

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

  // Iterate over all elements
  for (let i = 0; i < totalSize; i++) {
    // Convert linear index to multi-index
    let remaining = i;
    let condIdx = 0;
    let xIdx = 0;
    let yIdx = 0;

    for (let dim = 0; dim < maxNdim; dim++) {
      const idx = Math.floor(remaining / resultStrides[dim]!);
      remaining = remaining % resultStrides[dim]!;

      condIdx += idx * condStrides[dim]!;
      xIdx += idx * xStrides[dim]!;
      yIdx += idx * yStrides[dim]!;
    }

    if (isNonZero(condData, condIdx, isCondComplex)) {
      if (isResultComplex) {
        // Copy both real and imaginary parts
        (resultData as Float64Array | Float32Array)[i * 2] = (xData as Float64Array | Float32Array)[xIdx * 2]!;
        (resultData as Float64Array | Float32Array)[i * 2 + 1] = (xData as Float64Array | Float32Array)[xIdx * 2 + 1]!;
      } else {
        resultData[i] = xData[xIdx]!;
      }
    } else {
      if (isResultComplex) {
        // Copy both real and imaginary parts
        (resultData as Float64Array | Float32Array)[i * 2] = (yData as Float64Array | Float32Array)[yIdx * 2]!;
        (resultData as Float64Array | Float32Array)[i * 2 + 1] = (yData as Float64Array | Float32Array)[yIdx * 2 + 1]!;
      } else {
        resultData[i] = yData[yIdx]!;
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

  const data = storage.data;
  const n = storage.size;
  const valuesData = values.data;
  const numValues = values.size;

  // Create result array
  const result = ArrayStorage.zeros([numValues], 'int32');
  const resultData = result.data as Int32Array;

  // Binary search for each value
  for (let i = 0; i < numValues; i++) {
    const v = Number(valuesData[i]);
    let lo = 0;
    let hi = n;

    if (side === 'left') {
      // Find leftmost position where v can be inserted
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (Number(data[mid]) < v) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
    } else {
      // Find rightmost position where v can be inserted
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (Number(data[mid]) <= v) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
    }

    resultData[i] = lo;
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
  const condData = condition.data;
  const data = storage.data;
  const dtype = storage.dtype;
  const isCondComplex = isComplexDType(condition.dtype);
  const isDataComplex = isComplexDType(dtype);

  // Both arrays should have same size
  const size = Math.min(condition.size, storage.size);

  // Count number of true values
  let count = 0;
  for (let i = 0; i < size; i++) {
    if (isNonZero(condData, i, isCondComplex)) {
      count++;
    }
  }

  // Create result array
  const result = ArrayStorage.zeros([count], dtype as DType);
  const resultData = result.data;

  // Copy values where condition is true
  let idx = 0;
  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      if (isNonZero(condData, i, isCondComplex)) {
        resultTyped[idx++] = typedData[i]!;
      }
    }
  } else if (isDataComplex) {
    // Complex data: copy both real and imaginary parts
    const typedData = data as Float64Array | Float32Array;
    const resultTyped = resultData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      if (isNonZero(condData, i, isCondComplex)) {
        resultTyped[idx * 2] = typedData[i * 2]!;
        resultTyped[idx * 2 + 1] = typedData[i * 2 + 1]!;
        idx++;
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      if (isNonZero(condData, i, isCondComplex)) {
        resultData[idx++] = data[i]!;
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
  const isComplex = isComplexDType(storage.dtype);

  if (axis === undefined) {
    // Count all non-zero elements
    let count = 0;
    for (let i = 0; i < size; i++) {
      if (isNonZero(data, i, isComplex)) {
        count++;
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
      const linearIdx = multiIndexToLinear(inputIndices, shape);
      if (isNonZero(data, linearIdx, isComplex)) {
        count++;
      }
    }
    resultData[outerIdx] = count;
  }

  return result;
}
