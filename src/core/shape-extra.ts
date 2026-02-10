/**
 * Additional shape manipulation functions
 *
 * Tree-shakeable standalone functions for array manipulation.
 */

import { NDArrayCore, type DType } from '../core/ndarray-core';
import { array, zeros } from '../creation';
import { concatenate, flatten } from './shape';

/**
 * Append values to the end of an array
 */
export function append(
  arr: NDArrayCore,
  values: NDArrayCore | number | number[],
  axis?: number
): NDArrayCore {
  const valuesArr =
    values instanceof NDArrayCore ? values : array(Array.isArray(values) ? values : [values]);

  if (axis === undefined) {
    // Flatten both and concatenate
    const flatArr = flatten(arr);
    const flatValues = flatten(valuesArr);
    return concatenate([flatArr, flatValues], 0);
  }

  // Concatenate along axis
  return concatenate([arr, valuesArr], axis);
}

/**
 * Delete elements from an array
 */
export function delete_(arr: NDArrayCore, obj: number | number[], axis?: number): NDArrayCore {
  const indices = Array.isArray(obj) ? obj : [obj];
  const shape = [...arr.shape];
  const data = arr.data;

  if (axis === undefined) {
    // Flatten and delete
    const flat = flatten(arr);
    const flatData = flat.data;
    const newData: number[] = [];

    const deleteSet = new Set(indices.map((i) => (i < 0 ? flat.size + i : i)));

    for (let i = 0; i < flat.size; i++) {
      if (!deleteSet.has(i)) {
        newData.push(flatData[i] as number);
      }
    }

    return array(newData, arr.dtype as DType);
  }

  // Delete along axis
  const normalizedAxis = axis < 0 ? shape.length + axis : axis;
  const axisLen = shape[normalizedAxis]!;

  const deleteSet = new Set(indices.map((i) => (i < 0 ? axisLen + i : i)));

  // Calculate new shape
  const newAxisLen = axisLen - deleteSet.size;
  const newShape = [...shape];
  newShape[normalizedAxis] = newAxisLen;

  const result = zeros(newShape, arr.dtype as DType);
  const resultData = result.data;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  const newStrides: number[] = [];
  stride = 1;
  for (let i = newShape.length - 1; i >= 0; i--) {
    newStrides.unshift(stride);
    stride *= newShape[i]!;
  }

  // Copy data, skipping deleted indices
  const totalSize = shape.reduce((a, b) => a * b, 1);
  for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
    // Get multi-index
    const multiIdx: number[] = [];
    let remaining = flatIdx;
    for (let i = 0; i < shape.length; i++) {
      multiIdx.push(Math.floor(remaining / strides[i]!));
      remaining = remaining % strides[i]!;
    }

    // Skip if this index is being deleted
    if (deleteSet.has(multiIdx[normalizedAxis]!)) {
      continue;
    }

    // Calculate new index
    let deletedBefore = 0;
    for (const delIdx of deleteSet) {
      if (delIdx < multiIdx[normalizedAxis]!) {
        deletedBefore++;
      }
    }

    const newMultiIdx = [...multiIdx];
    (newMultiIdx[normalizedAxis] as number) -= deletedBefore;

    let newFlatIdx = 0;
    for (let i = 0; i < newShape.length; i++) {
      newFlatIdx += newMultiIdx[i]! * newStrides[i]!;
    }

    (resultData as Float64Array)[newFlatIdx] = data[flatIdx] as number;
  }

  return result;
}

// Export as both delete_ and delete (delete is reserved word but can be used as export name)
export { delete_ as delete };

/**
 * Insert values along an axis
 */
export function insert(
  arr: NDArrayCore,
  obj: number | number[],
  values: NDArrayCore | number | number[],
  axis?: number
): NDArrayCore {
  const indices = Array.isArray(obj) ? obj : [obj];
  const valuesArr =
    values instanceof NDArrayCore ? values : array(Array.isArray(values) ? values : [values]);

  if (axis === undefined) {
    // Flatten and insert
    const flat = flatten(arr);
    const flatValues = flatten(valuesArr);
    const flatData = flat.data;
    const valuesData = flatValues.data;

    // Sort indices in descending order to insert from end
    const sortedIndices = [...indices].sort((a, b) => {
      const ai = a < 0 ? flat.size + a : a;
      const bi = b < 0 ? flat.size + b : b;
      return ai - bi;
    });

    const result: number[] = Array.from(flatData as unknown as ArrayLike<number>);

    for (let i = 0; i < sortedIndices.length; i++) {
      const sortedIdx = sortedIndices[i]!;
      const idx = sortedIdx < 0 ? flat.size + sortedIdx : sortedIdx;
      const val = valuesData[i % valuesData.length] as number;
      result.splice(idx + i, 0, val);
    }

    return array(result, arr.dtype as DType);
  }

  // Insert along axis - simplified implementation
  throw new Error('insert along axis not fully implemented in standalone');
}

/**
 * Pad an array
 */
export function pad(
  arr: NDArrayCore,
  pad_width: number | [number, number] | [number, number][],
  mode:
    | 'constant'
    | 'edge'
    | 'linear_ramp'
    | 'maximum'
    | 'mean'
    | 'median'
    | 'minimum'
    | 'reflect'
    | 'symmetric'
    | 'wrap'
    | 'empty' = 'constant',
  constant_values: number = 0
): NDArrayCore {
  const shape = [...arr.shape];
  const ndim = shape.length;

  // Normalize pad_width to [[before, after], ...] for each dimension
  let padWidths: [number, number][];
  if (typeof pad_width === 'number') {
    padWidths = Array(ndim).fill([pad_width, pad_width]) as [number, number][];
  } else if (Array.isArray(pad_width) && typeof pad_width[0] === 'number') {
    padWidths = Array(ndim).fill(pad_width as [number, number]) as [number, number][];
  } else {
    padWidths = pad_width as [number, number][];
  }

  // Calculate new shape
  const newShape = shape.map((s, i) => s + padWidths[i]![0] + padWidths[i]![1]);

  if (mode !== 'constant') {
    throw new Error(`pad mode '${mode}' not fully implemented in standalone`);
  }

  // Create result array filled with constant value
  const result = zeros(newShape, arr.dtype as DType);
  const resultData = result.data;
  if (constant_values !== 0) {
    (resultData as Float64Array).fill(constant_values);
  }

  // Copy original data into the padded position
  const data = arr.data;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  const newStrides: number[] = [];
  stride = 1;
  for (let i = newShape.length - 1; i >= 0; i--) {
    newStrides.unshift(stride);
    stride *= newShape[i]!;
  }

  // Copy data
  const totalSize = shape.reduce((a, b) => a * b, 1);
  for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
    // Get multi-index in original array
    const multiIdx: number[] = [];
    let remaining = flatIdx;
    for (let i = 0; i < shape.length; i++) {
      multiIdx.push(Math.floor(remaining / strides[i]!));
      remaining = remaining % strides[i]!;
    }

    // Calculate new index with padding offset
    let newFlatIdx = 0;
    for (let i = 0; i < newShape.length; i++) {
      newFlatIdx += (multiIdx[i]! + padWidths[i]![0]) * newStrides[i]!;
    }

    (resultData as Float64Array)[newFlatIdx] = data[flatIdx] as number;
  }

  return result;
}
