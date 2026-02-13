/**
 * Shape manipulation operations
 *
 * Pure functions for reshaping, transposing, and manipulating array dimensions.
 * @module ops/shape
 */

import { ArrayStorage, computeStrides } from '../storage';
import { getTypedArrayConstructor, isBigIntDType, type TypedArray } from '../dtype';

/**
 * Reshape array to a new shape
 * Returns a view if array is C-contiguous, otherwise copies data
 */
export function reshape(storage: ArrayStorage, newShape: number[]): ArrayStorage {
  const size = storage.size;
  const dtype = storage.dtype;

  // Check if -1 is in the shape (infer dimension)
  const negIndex = newShape.indexOf(-1);
  let finalShape: number[];

  if (negIndex !== -1) {
    // Infer the dimension at negIndex
    const knownSize = newShape.reduce((acc, dim, i) => (i === negIndex ? acc : acc * dim), 1);
    const inferredDim = size / knownSize;

    if (!Number.isInteger(inferredDim)) {
      throw new Error(
        `cannot reshape array of size ${size} into shape ${JSON.stringify(newShape)}`
      );
    }

    finalShape = newShape.map((dim, i) => (i === negIndex ? inferredDim : dim));
  } else {
    finalShape = newShape;
  }

  // Validate that the new shape has the same total size
  const newSize = finalShape.reduce((a, b) => a * b, 1);
  if (newSize !== size) {
    throw new Error(
      `cannot reshape array of size ${size} into shape ${JSON.stringify(finalShape)}`
    );
  }

  // Fast path: if array is C-contiguous, create a view (no copy)
  if (storage.isCContiguous) {
    const data = storage.data;
    return ArrayStorage.fromData(data, finalShape, dtype, computeStrides(finalShape), 0);
  }

  // Slow path: array is not contiguous, must copy data first
  // Create contiguous copy, then reshape
  const contiguousCopy = storage.copy(); // copy() creates C-contiguous array
  const data = contiguousCopy.data;
  return ArrayStorage.fromData(data, finalShape, dtype, computeStrides(finalShape), 0);
}

/**
 * Return a flattened copy of the array
 * Creates 1D array containing all elements in row-major order
 * Always returns a copy (matching NumPy behavior)
 */
export function flatten(storage: ArrayStorage): ArrayStorage {
  const size = storage.size;
  const dtype = storage.dtype;
  const Constructor = getTypedArrayConstructor(dtype);

  if (!Constructor) {
    throw new Error(`Cannot flatten array with dtype ${dtype}`);
  }

  // Fast path: if array is C-contiguous, just copy the underlying data
  if (storage.isCContiguous) {
    const data = storage.data;
    const newData = data.slice(storage.offset, storage.offset + size);
    return ArrayStorage.fromData(newData as TypedArray, [size], dtype, [1], 0);
  }

  // Slow path: non-contiguous array, copy using iget (flat index)
  // This is much faster than recursive get(...indices) calls
  const newData = new Constructor(size);
  const isBigInt = isBigIntDType(dtype);

  for (let i = 0; i < size; i++) {
    const value = storage.iget(i);
    if (isBigInt) {
      (newData as BigInt64Array | BigUint64Array)[i] = value as bigint;
    } else {
      (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
    }
  }

  return ArrayStorage.fromData(newData, [size], dtype, [1], 0);
}

/**
 * Return a flattened array (view when possible, otherwise copy)
 * Returns a view if array is C-contiguous, otherwise copies data
 */
export function ravel(storage: ArrayStorage): ArrayStorage {
  const size = storage.size;
  const dtype = storage.dtype;

  // Fast path: if array is C-contiguous, create a view (no copy needed)
  if (storage.isCContiguous) {
    const data = storage.data;
    return ArrayStorage.fromData(data, [size], dtype, [1], 0);
  }

  // Slow path: array is not contiguous, must copy like flatten()
  return flatten(storage);
}

/**
 * Transpose array (permute dimensions)
 * Returns a view with transposed dimensions
 */
export function transpose(storage: ArrayStorage, axes?: number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  let permutation: number[];

  if (axes === undefined) {
    // Default: reverse all dimensions
    permutation = Array.from({ length: ndim }, (_, i) => ndim - 1 - i);
  } else {
    // Validate axes
    if (axes.length !== ndim) {
      throw new Error(`axes must have length ${ndim}, got ${axes.length}`);
    }

    // Check that axes is a valid permutation
    const seen = new Set<number>();
    for (const axis of axes) {
      const normalizedAxis = axis < 0 ? ndim + axis : axis;
      if (normalizedAxis < 0 || normalizedAxis >= ndim) {
        throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
      }
      if (seen.has(normalizedAxis)) {
        throw new Error(`repeated axis in transpose`);
      }
      seen.add(normalizedAxis);
    }

    permutation = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  // Compute new shape and strides
  const newShape = permutation.map((i) => shape[i]!);
  const oldStrides = Array.from(strides);
  const newStrides = permutation.map((i) => oldStrides[i]!);

  // Create transposed view
  return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
}

/**
 * Remove axes of length 1
 * Returns a view with specified dimensions removed
 */
export function squeeze(storage: ArrayStorage, axis?: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  if (axis === undefined) {
    // Remove all axes with size 1
    const newShape: number[] = [];
    const newStrides: number[] = [];

    for (let i = 0; i < ndim; i++) {
      if (shape[i] !== 1) {
        newShape.push(shape[i]!);
        newStrides.push(strides[i]!);
      }
    }

    // If all dimensions were 1, result would be a scalar (0-d array)
    // For now, keep at least one dimension
    if (newShape.length === 0) {
      newShape.push(1);
      newStrides.push(1);
    }

    return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
  } else {
    // Normalize axis
    const normalizedAxis = axis < 0 ? ndim + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Check that the axis has size 1
    if (shape[normalizedAxis] !== 1) {
      throw new Error(
        `cannot select an axis which has size not equal to one (axis ${axis} has size ${shape[normalizedAxis]})`
      );
    }

    // Remove the specified axis
    const newShape: number[] = [];
    const newStrides: number[] = [];

    for (let i = 0; i < ndim; i++) {
      if (i !== normalizedAxis) {
        newShape.push(shape[i]!);
        newStrides.push(strides[i]!);
      }
    }

    return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
  }
}

/**
 * Expand the shape by inserting a new axis of length 1
 * Returns a view with additional dimension
 */
export function expandDims(storage: ArrayStorage, axis: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  // Normalize axis (can be from -ndim-1 to ndim)
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + axis + 1;
  }

  if (normalizedAxis < 0 || normalizedAxis > ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim + 1}`);
  }

  // Insert 1 at the specified position
  const newShape = [...Array.from(shape)];
  newShape.splice(normalizedAxis, 0, 1);

  // Insert a stride at the new axis position
  // The stride for a dimension of size 1 doesn't matter, but conventionally
  // it should be the product of all dimensions to its right
  const newStrides = [...Array.from(strides)];
  const insertedStride =
    normalizedAxis < ndim ? strides[normalizedAxis]! * (shape[normalizedAxis] || 1) : 1;
  newStrides.splice(normalizedAxis, 0, insertedStride);

  return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
}

/**
 * Swap two axes of an array
 * Returns a view with axes swapped
 */
export function swapaxes(storage: ArrayStorage, axis1: number, axis2: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;
  const data = storage.data;
  const dtype = storage.dtype;

  // Normalize axes
  let normalizedAxis1 = axis1 < 0 ? ndim + axis1 : axis1;
  let normalizedAxis2 = axis2 < 0 ? ndim + axis2 : axis2;

  if (normalizedAxis1 < 0 || normalizedAxis1 >= ndim) {
    throw new Error(`axis1 ${axis1} is out of bounds for array of dimension ${ndim}`);
  }
  if (normalizedAxis2 < 0 || normalizedAxis2 >= ndim) {
    throw new Error(`axis2 ${axis2} is out of bounds for array of dimension ${ndim}`);
  }

  // If same axis, return a view without change
  if (normalizedAxis1 === normalizedAxis2) {
    return ArrayStorage.fromData(
      data,
      Array.from(shape),
      dtype,
      Array.from(strides),
      storage.offset
    );
  }

  // Swap shape and strides
  const newShape = Array.from(shape);
  const newStrides = Array.from(strides);

  [newShape[normalizedAxis1], newShape[normalizedAxis2]] = [
    newShape[normalizedAxis2]!,
    newShape[normalizedAxis1]!,
  ];
  [newStrides[normalizedAxis1], newStrides[normalizedAxis2]] = [
    newStrides[normalizedAxis2]!,
    newStrides[normalizedAxis1]!,
  ];

  return ArrayStorage.fromData(data, newShape, dtype, newStrides, storage.offset);
}

/**
 * Move axes to new positions
 * Returns a view with axes moved
 */
export function moveaxis(
  storage: ArrayStorage,
  source: number | number[],
  destination: number | number[]
): ArrayStorage {
  const ndim = storage.ndim;

  // Convert to arrays
  const sourceArr = Array.isArray(source) ? source : [source];
  const destArr = Array.isArray(destination) ? destination : [destination];

  if (sourceArr.length !== destArr.length) {
    throw new Error('source and destination must have the same number of elements');
  }

  // Normalize axes
  const normalizedSource = sourceArr.map((ax) => {
    const normalized = ax < 0 ? ndim + ax : ax;
    if (normalized < 0 || normalized >= ndim) {
      throw new Error(`source axis ${ax} is out of bounds for array of dimension ${ndim}`);
    }
    return normalized;
  });

  const normalizedDest = destArr.map((ax) => {
    const normalized = ax < 0 ? ndim + ax : ax;
    if (normalized < 0 || normalized >= ndim) {
      throw new Error(`destination axis ${ax} is out of bounds for array of dimension ${ndim}`);
    }
    return normalized;
  });

  // Check for duplicate source/dest axes
  if (new Set(normalizedSource).size !== normalizedSource.length) {
    throw new Error('repeated axis in source');
  }
  if (new Set(normalizedDest).size !== normalizedDest.length) {
    throw new Error('repeated axis in destination');
  }

  // Build permutation
  // Start with axes not in source
  const order: number[] = [];
  for (let i = 0; i < ndim; i++) {
    if (!normalizedSource.includes(i)) {
      order.push(i);
    }
  }

  // Insert source axes at destination positions
  for (let i = 0; i < normalizedSource.length; i++) {
    const dst = normalizedDest[i]!;
    order.splice(dst, 0, normalizedSource[i]!);
  }

  return transpose(storage, order);
}

/**
 * Concatenate arrays along an axis
 */
export function concatenate(storages: ArrayStorage[], axis: number = 0): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to concatenate');
  }

  if (storages.length === 1) {
    return storages[0]!.copy();
  }

  const first = storages[0]!;
  const ndim = first.ndim;
  const dtype = first.dtype;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Validate shapes: all arrays must have same shape except along axis
  for (let i = 1; i < storages.length; i++) {
    const s = storages[i]!;
    if (s.ndim !== ndim) {
      throw new Error('all the input arrays must have same number of dimensions');
    }
    for (let d = 0; d < ndim; d++) {
      if (d !== normalizedAxis && s.shape[d] !== first.shape[d]) {
        throw new Error(
          `all the input array dimensions except for the concatenation axis must match exactly`
        );
      }
    }
  }

  // Calculate output shape
  const outputShape = Array.from(first.shape);
  let totalAlongAxis = first.shape[normalizedAxis]!;
  for (let i = 1; i < storages.length; i++) {
    totalAlongAxis += storages[i]!.shape[normalizedAxis]!;
  }
  outputShape[normalizedAxis] = totalAlongAxis;

  // Create output array
  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot concatenate arrays with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // Copy data from each input array
  let offset = 0;
  for (const storage of storages) {
    const axisSize = storage.shape[normalizedAxis]!;
    copyToOutput(storage, outputData, outputShape, outputStrides, normalizedAxis, offset, dtype);
    offset += axisSize;
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Helper to copy data into concatenated output
 */
function copyToOutput(
  source: ArrayStorage,
  outputData: TypedArray,
  _outputShape: number[],
  outputStrides: number[],
  axis: number,
  axisOffset: number,
  dtype: string
): void {
  const sourceShape = source.shape;
  const ndim = sourceShape.length;
  const sourceSize = source.size;
  const isBigInt = dtype === 'int64' || dtype === 'uint64';

  // Fast path: if concatenating along axis 0 and both are C-contiguous,
  // we can do bulk copy
  if (axis === 0 && source.isCContiguous && ndim > 0) {
    // Calculate the starting position in output array
    const outputOffset = axisOffset * outputStrides[0]!;
    const sourceData = source.data;
    const start = source.offset;
    const end = start + sourceSize;

    // Bulk copy the entire source array
    // @ts-expect-error - TypedArray.set() works with any typed array subarray
    outputData.set(sourceData.subarray(start, end), outputOffset);
    return;
  }

  // Optimized path for axis=1 (common for hstack): copy row by row
  if (axis === 1 && ndim === 2 && source.isCContiguous) {
    const rows = sourceShape[0]!;
    const cols = sourceShape[1]!;
    const outputCols = _outputShape[1]!;
    const sourceData = source.data;
    const sourceStart = source.offset;

    for (let row = 0; row < rows; row++) {
      const sourceRowStart = sourceStart + row * cols;
      const outputRowStart = row * outputCols + axisOffset;
      // @ts-expect-error - TypedArray.set() works with any typed array subarray
      outputData.set(sourceData.subarray(sourceRowStart, sourceRowStart + cols), outputRowStart);
    }
    return;
  }

  // Slow path: element-by-element copy using flat indices (iget)
  // Optimized to avoid array spread and pre-compute base offset
  const indices = new Array(ndim).fill(0);
  const baseOutputOffset = axisOffset * outputStrides[axis]!;

  for (let i = 0; i < sourceSize; i++) {
    // Get value from source using flat index
    const value = source.iget(i);

    // Compute output index more efficiently
    let outputIdx = baseOutputOffset;
    for (let d = 0; d < ndim; d++) {
      outputIdx += indices[d]! * outputStrides[d]!;
    }

    // Write to output
    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[outputIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outputIdx] =
        value as number;
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d]! < sourceShape[d]!) {
        break;
      }
      indices[d] = 0;
    }
  }
}

/**
 * Stack arrays along a new axis
 */
export function stack(storages: ArrayStorage[], axis: number = 0): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  const first = storages[0]!;
  const shape = first.shape;
  const ndim = first.ndim;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + 1 + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis > ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim + 1}`);
  }

  // Validate shapes: all arrays must have exact same shape
  for (let i = 1; i < storages.length; i++) {
    const s = storages[i]!;
    if (s.ndim !== ndim) {
      throw new Error('all input arrays must have the same shape');
    }
    for (let d = 0; d < ndim; d++) {
      if (s.shape[d] !== shape[d]) {
        throw new Error('all input arrays must have the same shape');
      }
    }
  }

  // Expand dims on each array, then concatenate
  const expanded = storages.map((s) => expandDims(s, normalizedAxis));
  return concatenate(expanded, normalizedAxis);
}

/**
 * Stack arrays vertically (row-wise)
 * vstack is equivalent to concatenation along the first axis after
 * 1-D arrays of shape (N,) have been reshaped to (1,N)
 */
export function vstack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // For 1D arrays, reshape to (1, N) first
  const prepared = storages.map((s) => {
    if (s.ndim === 1) {
      return reshape(s, [1, s.shape[0]!]);
    }
    return s;
  });

  return concatenate(prepared, 0);
}

/**
 * Stack arrays horizontally (column-wise)
 * hstack is equivalent to concatenation along the second axis,
 * except for 1-D arrays where it concatenates along the first axis
 */
export function hstack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // Check if all arrays are 1D
  const allOneDim = storages.every((s) => s.ndim === 1);

  if (allOneDim) {
    // For 1D arrays, concatenate along axis 0
    return concatenate(storages, 0);
  }

  // For higher-dimensional arrays, concatenate along axis 1
  return concatenate(storages, 1);
}

/**
 * Stack arrays depth-wise (along third axis)
 * dstack is equivalent to concatenation along the third axis after
 * 2-D arrays of shape (M,N) have been reshaped to (M,N,1) and
 * 1-D arrays of shape (N,) have been reshaped to (1,N,1)
 */
export function dstack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // Prepare arrays
  const prepared = storages.map((s) => {
    if (s.ndim === 1) {
      // Reshape (N,) to (1, N, 1)
      return reshape(expandDims(reshape(s, [1, s.shape[0]!]), 2), [1, s.shape[0]!, 1]);
    } else if (s.ndim === 2) {
      // Reshape (M, N) to (M, N, 1)
      return expandDims(s, 2);
    }
    return s;
  });

  return concatenate(prepared, 2);
}

/**
 * Split array into sub-arrays
 */
export function split(
  storage: ArrayStorage,
  indicesOrSections: number | number[],
  axis: number = 0
): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  let splitIndices: number[];

  if (typeof indicesOrSections === 'number') {
    // Split into N equal sections
    if (axisSize % indicesOrSections !== 0) {
      throw new Error(`array split does not result in an equal division`);
    }
    const sectionSize = axisSize / indicesOrSections;
    splitIndices = [];
    for (let i = 1; i < indicesOrSections; i++) {
      splitIndices.push(i * sectionSize);
    }
  } else {
    // Split at specified indices
    splitIndices = indicesOrSections;
  }

  return splitAtIndices(storage, splitIndices, normalizedAxis);
}

/**
 * Split array into sub-arrays (allows unequal splits)
 */
export function arraySplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[],
  axis: number = 0
): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  let splitIndices: number[];

  if (typeof indicesOrSections === 'number') {
    // Split into N sections (may be unequal)
    const numSections = indicesOrSections;
    const sectionSize = Math.floor(axisSize / numSections);
    const remainder = axisSize % numSections;

    splitIndices = [];
    let offset = 0;
    for (let i = 0; i < numSections - 1; i++) {
      offset += sectionSize + (i < remainder ? 1 : 0);
      splitIndices.push(offset);
    }
  } else {
    // Split at specified indices
    splitIndices = indicesOrSections;
  }

  return splitAtIndices(storage, splitIndices, normalizedAxis);
}

/**
 * Helper to split array at specified indices
 */
function splitAtIndices(storage: ArrayStorage, indices: number[], axis: number): ArrayStorage[] {
  const shape = storage.shape;
  const axisSize = shape[axis]!;

  // Add boundaries
  const boundaries = [0, ...indices, axisSize];
  const result: ArrayStorage[] = [];

  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i]!;
    const end = boundaries[i + 1]!;

    if (start > end) {
      throw new Error('split indices must be in ascending order');
    }

    // Create slice
    const sliceShape = Array.from(shape);
    sliceShape[axis] = end - start;

    // Calculate new offset and strides
    const newOffset = storage.offset + start * storage.strides[axis]!;

    result.push(
      ArrayStorage.fromData(
        storage.data,
        sliceShape,
        storage.dtype,
        Array.from(storage.strides),
        newOffset
      )
    );
  }

  return result;
}

/**
 * Split array vertically (row-wise)
 */
export function vsplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[]
): ArrayStorage[] {
  if (storage.ndim < 2) {
    throw new Error('vsplit only works on arrays of 2 or more dimensions');
  }
  return arraySplit(storage, indicesOrSections, 0);
}

/**
 * Split array horizontally (column-wise)
 */
export function hsplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[]
): ArrayStorage[] {
  if (storage.ndim < 1) {
    throw new Error('hsplit only works on arrays of 1 or more dimensions');
  }
  // For 1D arrays, split along axis 0; for higher dims, split along axis 1
  const axis = storage.ndim === 1 ? 0 : 1;
  return arraySplit(storage, indicesOrSections, axis);
}

/**
 * Tile array by repeating along each axis
 */
export function tile(storage: ArrayStorage, reps: number | number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  // Normalize reps to array
  const repsArr = Array.isArray(reps) ? reps : [reps];

  // Pad reps or shape to match dimensions
  const maxDim = Math.max(ndim, repsArr.length);
  const paddedShape = new Array(maxDim).fill(1);
  const paddedReps = new Array(maxDim).fill(1);

  // Fill from the right
  for (let i = 0; i < ndim; i++) {
    paddedShape[maxDim - ndim + i] = shape[i]!;
  }
  for (let i = 0; i < repsArr.length; i++) {
    paddedReps[maxDim - repsArr.length + i] = repsArr[i]!;
  }

  // Calculate output shape
  const outputShape = paddedShape.map((s, i) => s * paddedReps[i]!);
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot tile array with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // If we need to expand dimensions of input, reshape it
  let expandedStorage = storage;
  if (ndim < maxDim) {
    expandedStorage = reshape(storage, paddedShape);
  }

  const isBigInt = dtype === 'int64' || dtype === 'uint64';
  const expandedStrides = expandedStorage.strides;

  // Fill output by iterating through all output positions
  const outputIndices = new Array(maxDim).fill(0);

  for (let i = 0; i < outputSize; i++) {
    // Compute source flat index directly (wrap around for tiling)
    let sourceFlatIdx = expandedStorage.offset;
    for (let d = 0; d < maxDim; d++) {
      const sourceIdx = outputIndices[d]! % paddedShape[d]!;
      sourceFlatIdx += sourceIdx * expandedStrides[d]!;
    }

    // Read value directly from flat index
    const value = expandedStorage.data[sourceFlatIdx];

    // Write to output using direct index calculation
    let outputIdx = 0;
    for (let d = 0; d < maxDim; d++) {
      outputIdx += outputIndices[d]! * outputStrides[d]!;
    }

    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[outputIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outputIdx] =
        value as number;
    }

    // Increment indices
    for (let d = maxDim - 1; d >= 0; d--) {
      outputIndices[d]++;
      if (outputIndices[d]! < outputShape[d]!) {
        break;
      }
      outputIndices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Repeat elements of an array
 */
export function repeat(
  storage: ArrayStorage,
  repeats: number | number[],
  axis?: number
): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const size = storage.size;

  if (axis === undefined) {
    // Flatten and repeat each element
    const flatSize = size;
    const repeatsArr = Array.isArray(repeats) ? repeats : new Array(flatSize).fill(repeats);

    if (repeatsArr.length !== flatSize) {
      throw new Error(
        `operands could not be broadcast together with shape (${flatSize},) (${repeatsArr.length},)`
      );
    }

    const outputSize = repeatsArr.reduce((a, b) => a + b, 0);
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot repeat array with dtype ${dtype}`);
    }
    const outputData = new Constructor(outputSize);

    let outIdx = 0;
    for (let i = 0; i < flatSize; i++) {
      const value = storage.iget(i);
      const rep = repeatsArr[i]!;
      for (let r = 0; r < rep; r++) {
        if (dtype === 'int64' || dtype === 'uint64') {
          (outputData as BigInt64Array | BigUint64Array)[outIdx++] = value as bigint;
        } else {
          (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx++] =
            value as number;
        }
      }
    }

    return ArrayStorage.fromData(outputData, [outputSize], dtype);
  }

  // Repeat along specified axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const repeatsArr = Array.isArray(repeats) ? repeats : new Array(axisSize).fill(repeats);

  if (repeatsArr.length !== axisSize) {
    throw new Error(
      `operands could not be broadcast together with shape (${axisSize},) (${repeatsArr.length},)`
    );
  }

  // Calculate output shape
  const outputShape = Array.from(shape);
  outputShape[normalizedAxis] = repeatsArr.reduce((a, b) => a + b, 0);

  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot repeat array with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // Iterate through source and write repeated values
  const sourceIndices = new Array(ndim).fill(0);
  const isBigInt = dtype === 'int64' || dtype === 'uint64';

  // Track cumulative positions along axis
  const axisPositions: number[] = [0];
  for (let i = 0; i < axisSize; i++) {
    axisPositions.push(axisPositions[i]! + repeatsArr[i]!);
  }

  for (let i = 0; i < size; i++) {
    // Use iget for flat index access instead of get(...indices)
    const value = storage.iget(i);
    const axisIdx = sourceIndices[normalizedAxis]!;
    const rep = repeatsArr[axisIdx]!;

    // Calculate base output index (without axis component)
    let baseOutIdx = 0;
    for (let d = 0; d < ndim; d++) {
      if (d !== normalizedAxis) {
        baseOutIdx += sourceIndices[d]! * outputStrides[d]!;
      }
    }

    // Write repeated values
    const axisStride = outputStrides[normalizedAxis]!;
    const axisStart = axisPositions[axisIdx]!;
    for (let r = 0; r < rep; r++) {
      const outIdx = baseOutIdx + (axisStart + r) * axisStride;

      if (isBigInt) {
        (outputData as BigInt64Array | BigUint64Array)[outIdx] = value as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx] =
          value as number;
      }
    }

    // Increment source indices
    for (let d = ndim - 1; d >= 0; d--) {
      sourceIndices[d]++;
      if (sourceIndices[d]! < shape[d]!) {
        break;
      }
      sourceIndices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Reverse the order of elements in an array along given axis
 */
export function flip(storage: ArrayStorage, axis?: number | number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const size = storage.size;

  // Determine which axes to flip
  let axesToFlip: Set<number>;
  if (axis === undefined) {
    // Flip all axes
    axesToFlip = new Set(Array.from({ length: ndim }, (_, i) => i));
  } else if (typeof axis === 'number') {
    const normalizedAxis = axis < 0 ? ndim + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }
    axesToFlip = new Set([normalizedAxis]);
  } else {
    axesToFlip = new Set(
      axis.map((ax) => {
        const normalized = ax < 0 ? ndim + ax : ax;
        if (normalized < 0 || normalized >= ndim) {
          throw new Error(`axis ${ax} is out of bounds for array of dimension ${ndim}`);
        }
        return normalized;
      })
    );
  }

  // Create output array
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot flip array with dtype ${dtype}`);
  }
  const outputData = new Constructor(size);
  const isBigInt = isBigIntDType(dtype);

  // Fast path for 1D arrays
  if (ndim === 1 && storage.isCContiguous) {
    const sourceData = storage.data;
    const start = storage.offset;
    for (let i = 0; i < size; i++) {
      if (isBigInt) {
        (outputData as BigInt64Array | BigUint64Array)[i] = sourceData[
          start + size - 1 - i
        ] as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = sourceData[
          start + size - 1 - i
        ] as number;
      }
    }
    return ArrayStorage.fromData(outputData, [...shape], dtype);
  }

  // Fast path for 2D arrays
  if (ndim === 2 && storage.isCContiguous) {
    const rows = shape[0]!;
    const cols = shape[1]!;
    const sourceData = storage.data;
    const start = storage.offset;

    // Flipping both axes - reverse entire array
    if (axesToFlip.size === 2) {
      for (let i = 0; i < size; i++) {
        if (isBigInt) {
          (outputData as BigInt64Array | BigUint64Array)[i] = sourceData[
            start + size - 1 - i
          ] as bigint;
        } else {
          (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = sourceData[
            start + size - 1 - i
          ] as number;
        }
      }
      return ArrayStorage.fromData(outputData, [...shape], dtype);
    }

    if (axesToFlip.size === 1) {
      if (axesToFlip.has(0)) {
        // Flip rows: copy rows in reverse order
        for (let r = 0; r < rows; r++) {
          const sourceRowStart = start + (rows - 1 - r) * cols;
          const outputRowStart = r * cols;
          for (let c = 0; c < cols; c++) {
            if (isBigInt) {
              (outputData as BigInt64Array | BigUint64Array)[outputRowStart + c] = sourceData[
                sourceRowStart + c
              ] as bigint;
            } else {
              (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[
                outputRowStart + c
              ] = sourceData[sourceRowStart + c] as number;
            }
          }
        }
        return ArrayStorage.fromData(outputData, [...shape], dtype);
      } else if (axesToFlip.has(1)) {
        // Flip columns: reverse each row
        for (let r = 0; r < rows; r++) {
          const sourceRowStart = start + r * cols;
          const outputRowStart = r * cols;
          for (let c = 0; c < cols; c++) {
            if (isBigInt) {
              (outputData as BigInt64Array | BigUint64Array)[outputRowStart + c] = sourceData[
                sourceRowStart + cols - 1 - c
              ] as bigint;
            } else {
              (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[
                outputRowStart + c
              ] = sourceData[sourceRowStart + cols - 1 - c] as number;
            }
          }
        }
        return ArrayStorage.fromData(outputData, [...shape], dtype);
      }
    }
  }

  // General path: element-by-element copy
  const sourceIndices = new Array(ndim);
  const outputIndices = new Array(ndim).fill(0);

  for (let i = 0; i < size; i++) {
    // Compute source indices by flipping the specified axes
    for (let d = 0; d < ndim; d++) {
      sourceIndices[d] = axesToFlip.has(d) ? shape[d]! - 1 - outputIndices[d]! : outputIndices[d];
    }

    // Get value from source
    let sourceOffset = storage.offset;
    for (let d = 0; d < ndim; d++) {
      sourceOffset += sourceIndices[d]! * storage.strides[d]!;
    }
    const value = storage.data[sourceOffset];

    // Write to output
    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
    }

    // Increment output indices
    for (let d = ndim - 1; d >= 0; d--) {
      outputIndices[d]++;
      if (outputIndices[d]! < shape[d]!) {
        break;
      }
      outputIndices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, [...shape], dtype);
}

/**
 * Rotate array by 90 degrees in the plane specified by axes
 */
export function rot90(
  storage: ArrayStorage,
  k: number = 1,
  axes: [number, number] = [0, 1]
): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  if (ndim < 2) {
    throw new Error('Input must be at least 2-D');
  }

  // Normalize axes
  const axis0 = axes[0] < 0 ? ndim + axes[0] : axes[0];
  const axis1 = axes[1] < 0 ? ndim + axes[1] : axes[1];

  if (axis0 < 0 || axis0 >= ndim || axis1 < 0 || axis1 >= ndim) {
    throw new Error(`Axes are out of bounds for array of dimension ${ndim}`);
  }

  if (axis0 === axis1) {
    throw new Error('Axes must be different');
  }

  // Normalize k to 0-3
  k = ((k % 4) + 4) % 4;

  if (k === 0) {
    return storage.copy();
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot rotate array with dtype ${dtype}`);
  }

  const outputShape = [...shape];
  if (k === 1 || k === 3) {
    // Swap dimensions for axis0 and axis1
    [outputShape[axis0], outputShape[axis1]] = [outputShape[axis1]!, outputShape[axis0]!];
  }

  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const outputData = new Constructor(outputSize);
  const isBigInt = isBigIntDType(dtype);
  const srcData = storage.data;

  // Optimized 2D case (most common)
  if (ndim === 2 && axis0 === 0 && axis1 === 1) {
    const rows = shape[0]!;
    const cols = shape[1]!;

    if (k === 1) {
      // 90 degrees CCW: dst[outRow, outCol] = src[outCol, cols - 1 - outRow]
      // output shape: [cols, rows]
      const outRows = cols;
      const outCols = rows;
      if (isBigInt) {
        const src = srcData as BigInt64Array | BigUint64Array;
        const dst = outputData as BigInt64Array | BigUint64Array;
        for (let outRow = 0; outRow < outRows; outRow++) {
          const dstRowOffset = outRow * outCols;
          const srcCol = cols - 1 - outRow;
          for (let outCol = 0; outCol < outCols; outCol++) {
            dst[dstRowOffset + outCol] = src[outCol * cols + srcCol]!;
          }
        }
      } else {
        const src = srcData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        const dst = outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        for (let outRow = 0; outRow < outRows; outRow++) {
          const dstRowOffset = outRow * outCols;
          const srcCol = cols - 1 - outRow;
          for (let outCol = 0; outCol < outCols; outCol++) {
            dst[dstRowOffset + outCol] = src[outCol * cols + srcCol]!;
          }
        }
      }
    } else if (k === 2) {
      // 180 degrees: dst[outRow, outCol] = src[rows - 1 - outRow, cols - 1 - outCol]
      // output shape: [rows, cols]
      if (isBigInt) {
        const src = srcData as BigInt64Array | BigUint64Array;
        const dst = outputData as BigInt64Array | BigUint64Array;
        for (let outRow = 0; outRow < rows; outRow++) {
          const dstRowOffset = outRow * cols;
          const srcRowOffset = (rows - 1 - outRow) * cols;
          for (let outCol = 0; outCol < cols; outCol++) {
            dst[dstRowOffset + outCol] = src[srcRowOffset + (cols - 1 - outCol)]!;
          }
        }
      } else {
        const src = srcData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        const dst = outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        for (let outRow = 0; outRow < rows; outRow++) {
          const dstRowOffset = outRow * cols;
          const srcRowOffset = (rows - 1 - outRow) * cols;
          for (let outCol = 0; outCol < cols; outCol++) {
            dst[dstRowOffset + outCol] = src[srcRowOffset + (cols - 1 - outCol)]!;
          }
        }
      }
    } else {
      // k === 3, 270 degrees CCW (or 90 CW): dst[outRow, outCol] = src[rows - 1 - outCol, outRow]
      // output shape: [cols, rows]
      const outRows = cols;
      const outCols = rows;
      if (isBigInt) {
        const src = srcData as BigInt64Array | BigUint64Array;
        const dst = outputData as BigInt64Array | BigUint64Array;
        for (let outRow = 0; outRow < outRows; outRow++) {
          const dstRowOffset = outRow * outCols;
          for (let outCol = 0; outCol < outCols; outCol++) {
            dst[dstRowOffset + outCol] = src[(rows - 1 - outCol) * cols + outRow]!;
          }
        }
      } else {
        const src = srcData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        const dst = outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        for (let outRow = 0; outRow < outRows; outRow++) {
          const dstRowOffset = outRow * outCols;
          for (let outCol = 0; outCol < outCols; outCol++) {
            dst[dstRowOffset + outCol] = src[(rows - 1 - outCol) * cols + outRow]!;
          }
        }
      }
    }

    return ArrayStorage.fromData(outputData, outputShape, dtype);
  }

  // General N-D case (fallback)
  const outputStrides = computeStrides(outputShape);
  const inputStrides = computeStrides(shape);
  const indices = new Array(ndim).fill(0);
  const sourceIndices = new Array(ndim);

  for (let i = 0; i < storage.size; i++) {
    // Transform indices based on k
    for (let d = 0; d < ndim; d++) {
      sourceIndices[d] = indices[d];
    }

    let outIdx0, outIdx1;
    if (k === 1) {
      outIdx0 = shape[axis1]! - 1 - indices[axis1]!;
      outIdx1 = indices[axis0];
    } else if (k === 2) {
      outIdx0 = shape[axis0]! - 1 - indices[axis0]!;
      outIdx1 = shape[axis1]! - 1 - indices[axis1]!;
      sourceIndices[axis0] = outIdx0;
      sourceIndices[axis1] = outIdx1;
    } else {
      outIdx0 = indices[axis1];
      outIdx1 = shape[axis0]! - 1 - indices[axis0]!;
    }

    if (k !== 2) {
      sourceIndices[axis0] = outIdx0;
      sourceIndices[axis1] = outIdx1;
    }

    // Compute output offset
    let outputOffset = 0;
    for (let d = 0; d < ndim; d++) {
      outputOffset += sourceIndices[d]! * outputStrides[d]!;
    }

    // Get source value using direct typed array access
    let inputOffset = 0;
    for (let d = 0; d < ndim; d++) {
      inputOffset += indices[d]! * inputStrides[d]!;
    }

    // Write to output
    if (isBigInt) {
      const src = srcData as BigInt64Array | BigUint64Array;
      const dst = outputData as BigInt64Array | BigUint64Array;
      dst[outputOffset] = src[inputOffset]!;
    } else {
      const src = srcData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      const dst = outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      dst[outputOffset] = src[inputOffset]!;
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d]! < shape[d]!) {
        break;
      }
      indices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Roll array elements along a given axis
 */
export function roll(
  storage: ArrayStorage,
  shift: number | number[],
  axis?: number | number[]
): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  const size = storage.size;

  // Handle no axis case - roll as flattened array
  if (axis === undefined) {
    const flatShift = Array.isArray(shift) ? shift.reduce((a, b) => a + b, 0) : shift;
    const flatStorage = flatten(storage);

    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot roll array with dtype ${dtype}`);
    }
    const outputData = new Constructor(size);
    const isBigInt = isBigIntDType(dtype);

    for (let i = 0; i < size; i++) {
      const sourceIdx = (((i - flatShift) % size) + size) % size;
      const value = flatStorage.iget(sourceIdx);
      if (isBigInt) {
        (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
      }
    }

    // Reshape back to original shape
    return ArrayStorage.fromData(outputData, [...shape], dtype);
  }

  // Handle axis specified case
  const shifts = Array.isArray(shift) ? shift : [shift];
  const axes = Array.isArray(axis) ? axis : [axis];

  if (shifts.length !== axes.length) {
    throw new Error('shift and axis must have the same length');
  }

  // Normalize axes
  const normalizedAxes = axes.map((ax) => {
    const normalized = ax < 0 ? ndim + ax : ax;
    if (normalized < 0 || normalized >= ndim) {
      throw new Error(`axis ${ax} is out of bounds for array of dimension ${ndim}`);
    }
    return normalized;
  });

  // Create output array
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot roll array with dtype ${dtype}`);
  }
  const outputData = new Constructor(size);
  const isBigInt = isBigIntDType(dtype);

  // Iterate through all output positions
  const outputIndices = new Array(ndim).fill(0);

  for (let i = 0; i < size; i++) {
    // Compute source indices by rolling back
    const sourceIndices = [...outputIndices];
    for (let j = 0; j < normalizedAxes.length; j++) {
      const ax = normalizedAxes[j]!;
      const axisSize = shape[ax]!;
      const sh = shifts[j]!;
      sourceIndices[ax] = (((sourceIndices[ax]! - sh) % axisSize) + axisSize) % axisSize;
    }

    // Get value from source
    let sourceOffset = storage.offset;
    for (let d = 0; d < ndim; d++) {
      sourceOffset += sourceIndices[d]! * storage.strides[d]!;
    }
    const value = storage.data[sourceOffset];

    // Write to output
    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
    }

    // Increment output indices
    for (let d = ndim - 1; d >= 0; d--) {
      outputIndices[d]++;
      if (outputIndices[d]! < shape[d]!) {
        break;
      }
      outputIndices[d] = 0;
    }
  }

  return ArrayStorage.fromData(outputData, [...shape], dtype);
}

/**
 * Roll the specified axis backwards until it lies in a given position
 */
export function rollaxis(storage: ArrayStorage, axis: number, start: number = 0): ArrayStorage {
  const ndim = storage.ndim;

  // Normalize axis
  let normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Normalize start
  let normalizedStart = start < 0 ? ndim + start : start;
  if (normalizedStart < 0 || normalizedStart > ndim) {
    throw new Error(`start ${start} is out of bounds`);
  }

  if (normalizedAxis < normalizedStart) {
    normalizedStart--;
  }

  if (normalizedAxis === normalizedStart) {
    return ArrayStorage.fromData(
      storage.data,
      Array.from(storage.shape),
      storage.dtype,
      Array.from(storage.strides),
      storage.offset
    );
  }

  return moveaxis(storage, normalizedAxis, normalizedStart);
}

/**
 * Split array along third axis (depth)
 */
export function dsplit(
  storage: ArrayStorage,
  indicesOrSections: number | number[]
): ArrayStorage[] {
  if (storage.ndim < 3) {
    throw new Error('dsplit only works on arrays of 3 or more dimensions');
  }
  return arraySplit(storage, indicesOrSections, 2);
}

/**
 * Stack 1-D arrays as columns into a 2-D array
 */
export function columnStack(storages: ArrayStorage[]): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to stack');
  }

  // If all arrays are 1D, reshape them to column vectors
  const prepared = storages.map((s) => {
    if (s.ndim === 1) {
      return reshape(s, [s.shape[0]!, 1]);
    }
    return s;
  });

  return hstack(prepared);
}

/**
 * Stack arrays in sequence vertically (row_stack is an alias for vstack)
 */
export const rowStack = vstack;

/**
 * Resize array to new shape (returns new array, may repeat or truncate)
 */
export function resize(storage: ArrayStorage, newShape: number[]): ArrayStorage {
  const dtype = storage.dtype;
  const newSize = newShape.reduce((a, b) => a * b, 1);
  const oldSize = storage.size;

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot resize array with dtype ${dtype}`);
  }
  const outputData = new Constructor(newSize);
  const isBigInt = isBigIntDType(dtype);

  // Fill output by cycling through source data
  for (let i = 0; i < newSize; i++) {
    const sourceIdx = i % oldSize;
    const value = storage.iget(sourceIdx);
    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
    }
  }

  return ArrayStorage.fromData(outputData, newShape, dtype);
}

/**
 * Convert arrays to at least 1D
 */
export function atleast1d(storages: ArrayStorage[]): ArrayStorage[] {
  return storages.map((s) => {
    if (s.ndim === 0) {
      // 0-D array -> 1-D with one element
      return reshape(s, [1]);
    }
    return s;
  });
}

/**
 * Convert arrays to at least 2D
 */
export function atleast2d(storages: ArrayStorage[]): ArrayStorage[] {
  return storages.map((s) => {
    if (s.ndim === 0) {
      return reshape(s, [1, 1]);
    } else if (s.ndim === 1) {
      return reshape(s, [1, s.shape[0]!]);
    }
    return s;
  });
}

/**
 * Convert arrays to at least 3D
 */
export function atleast3d(storages: ArrayStorage[]): ArrayStorage[] {
  return storages.map((s) => {
    if (s.ndim === 0) {
      return reshape(s, [1, 1, 1]);
    } else if (s.ndim === 1) {
      return reshape(s, [1, s.shape[0]!, 1]);
    } else if (s.ndim === 2) {
      return reshape(s, [s.shape[0]!, s.shape[1]!, 1]);
    }
    return s;
  });
}

/**
 * Alias for concatenate
 */
export function concat(storages: ArrayStorage[], axis: number = 0): ArrayStorage {
  return concatenate(storages, axis);
}

/**
 * Split an array into a sequence of sub-arrays along an axis (inverse of stack)
 */
export function unstack(storage: ArrayStorage, axis: number = 0): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const results: ArrayStorage[] = [];

  for (let i = 0; i < axisSize; i++) {
    const slices: Array<{ start: number; stop: number; step: number }> = [];
    for (let d = 0; d < ndim; d++) {
      if (d === normalizedAxis) {
        slices.push({ start: i, stop: i + 1, step: 1 });
      } else {
        slices.push({ start: 0, stop: shape[d]!, step: 1 });
      }
    }
    const sliced = sliceStorage(storage, slices);
    const squeezed = squeeze(sliced, normalizedAxis);
    results.push(squeezed);
  }
  return results;
}

/**
 * Assemble an nd-array from nested lists of blocks
 * For a simple list [a, b] of nD arrays, concatenates along the last axis (like np.block)
 */
export function block(storages: ArrayStorage[], _depth: number = 1): ArrayStorage {
  if (storages.length === 0) {
    throw new Error('need at least one array to block');
  }
  if (storages.length === 1) {
    return storages[0]!.copy();
  }
  // np.block([a, b]) for nD arrays concatenates along the last axis (-1)
  // This matches NumPy's behavior where a flat list of arrays is joined horizontally
  return concatenate(storages, -1);
}

/**
 * Helper function to slice storage (used by unstack)
 */
function sliceStorage(
  storage: ArrayStorage,
  slices: Array<{ start: number; stop: number; step: number }>
): ArrayStorage {
  const shape = storage.shape;
  const strides = storage.strides;
  let offset = storage.offset;
  const dtype = storage.dtype;
  const data = storage.data;

  const newShape: number[] = [];
  const newStrides: number[] = [];

  for (let d = 0; d < shape.length; d++) {
    const slice = slices[d]!;
    const { start, stop, step } = slice;
    const dimSize = Math.ceil((stop - start) / step);
    newShape.push(dimSize);
    newStrides.push(strides[d]! * step);
    offset += start * strides[d]!;
  }

  return ArrayStorage.fromData(data, newShape, dtype, newStrides, offset);
}
