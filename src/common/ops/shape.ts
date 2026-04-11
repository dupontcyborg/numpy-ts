/**
 * Shape manipulation operations
 *
 * Pure functions for reshaping, transposing, and manipulating array dimensions.
 * @module ops/shape
 */

import { ArrayStorage, computeStrides } from '../storage';
import {
  getTypedArrayConstructor,
  isBigIntDType,
  isComplexDType,
  hasFloat16,
  type TypedArray,
} from '../dtype';
import { Complex } from '../complex';
import { wasmTile2D } from '../wasm/tile';
import { wasmRoll } from '../wasm/roll';
import { wasmRepeat } from '../wasm/repeat';
import { parseSlice, normalizeSlice } from '../slicing';
import { expandEllipsis } from '../internal/indexing';

/**
 * Slice an array along one or more dimensions
 * Returns a view (no data copy) with adjusted shape, strides, and offset
 */
export function slice(storage: ArrayStorage, ...slices: (string | number)[]): ArrayStorage {
  if (slices.length === 0) return storage;

  slices = expandEllipsis(slices, storage.ndim);

  const newShape: number[] = [];
  const newStrides: number[] = [];
  let newOffset = storage.offset;

  let axis = 0;
  for (let i = 0; i < slices.length; i++) {
    const slice = slices[i]!;
    if (typeof slice === 'number') {
      const start = slice;
      // Like normalizeSize, need to handle negative / out of bounds indices.
      const size = storage.shape[axis]!;
      const normalizedStart = start < 0 ? size + start : start;
      if (normalizedStart < 0 || normalizedStart >= size) {
        throw new Error(`Index ${start} is out of bounds for size ${size}`);
      }
      const stride = storage.strides[axis]!;
      axis++;
      newOffset += normalizedStart * stride;
      continue;
    } else if (slice === 'newaxis') {
      newShape.push(1);
      newStrides.push(0);
    } else {
      const parsed = parseSlice(slice);
      const spec = normalizeSlice(parsed, storage.shape[axis]!);
      const stride = storage.strides[axis]!;
      axis++;
      newOffset += spec.start * stride;

      if (spec.isIndex) continue;

      const sliceLength = Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
      newShape.push(sliceLength);
      newStrides.push(stride * spec.step);
    }
  }

  return ArrayStorage.fromDataShared(
    storage.data,
    newShape,
    storage.dtype,
    newStrides,
    newOffset,
    storage.wasmRegion
  );
}

/**
 * Slice an array along one or more dimensions, keeping all dimensions (rank-preserving)
 * Like slice(), but integer-index specs produce a size-1 dimension instead of dropping it.
 * Returns a view (no data copy) with adjusted shape, strides, and offset
 */
export function sliceKeepDim(storage: ArrayStorage, ...slices: string[]): ArrayStorage {
  if (slices.length === 0) return storage;

  slices = expandEllipsis(slices, storage.ndim);

  const newShape: number[] = [];
  const newStrides: number[] = [];
  let newOffset = storage.offset;

  let axis = 0;
  for (let i = 0; i < slices.length; i++) {
    const slice = slices[i]!;
    if (typeof slice === 'number') {
      const stride = storage.strides[axis]!;
      axis++;
      newOffset += slice * stride;
      newShape.push(1);
      newStrides.push(stride);
    } else if (slice === 'newaxis') {
      newShape.push(1);
      newStrides.push(0);
    } else {
      const parsed = parseSlice(slice);
      const spec = normalizeSlice(parsed, storage.shape[axis]!);
      const stride = storage.strides[axis]!;
      axis++;
      newOffset += spec.start * stride;

      // Unlike slice(), isIndex dimensions are kept as size-1 (rank is preserved)
      const sliceLength = spec.isIndex
        ? 1
        : Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
      newShape.push(sliceLength);
      newStrides.push(stride * spec.step);
    }
  }

  return ArrayStorage.fromDataShared(
    storage.data,
    newShape,
    storage.dtype,
    newStrides,
    newOffset,
    storage.wasmRegion
  );
}

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
    return ArrayStorage.fromDataShared(
      data,
      finalShape,
      dtype,
      computeStrides(finalShape),
      0,
      storage.wasmRegion
    );
  }

  // Slow path: array is not contiguous, must copy data into new shape
  const isComplex = isComplexDType(dtype);
  const result = ArrayStorage.empty(finalShape, dtype);
  const resultData = result.data;
  const isBigInt = isBigIntDType(dtype);
  if (isComplex) {
    for (let i = 0; i < size; i++) {
      const value = storage.iget(i) as Complex;
      (resultData as Float64Array | Float32Array)[i * 2] = value.re;
      (resultData as Float64Array | Float32Array)[i * 2 + 1] = value.im;
    }
  } else if (isBigInt) {
    for (let i = 0; i < size; i++) {
      (resultData as BigInt64Array | BigUint64Array)[i] = storage.iget(i) as bigint;
    }
  } else {
    for (let i = 0; i < size; i++) {
      (resultData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = storage.iget(
        i
      ) as number;
    }
  }
  return result;
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

  const isComplex = isComplexDType(dtype);
  const physicalSize = isComplex ? size * 2 : size;

  // Fast path: if array is C-contiguous, just copy the underlying data
  if (storage.isCContiguous) {
    const data = storage.data;
    const physicalOffset = isComplex ? storage.offset * 2 : storage.offset;
    const result = ArrayStorage.empty([size], dtype);
    (result.data as unknown as { set(src: ArrayLike<number>, offset?: number): void }).set(
      data.subarray(physicalOffset, physicalOffset + physicalSize) as unknown as ArrayLike<number>
    );
    return result;
  }

  // Slow path: non-contiguous array, copy using iget (flat index)
  // This is much faster than recursive get(...indices) calls
  const result = ArrayStorage.empty([size], dtype);
  const newData = result.data;
  const isBigInt = isBigIntDType(dtype);

  if (isComplex) {
    for (let i = 0; i < size; i++) {
      const value = storage.iget(i) as Complex;
      (newData as Float64Array | Float32Array)[i * 2] = value.re;
      (newData as Float64Array | Float32Array)[i * 2 + 1] = value.im;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const value = storage.iget(i);
      if (isBigInt) {
        (newData as BigInt64Array | BigUint64Array)[i] = value as bigint;
      } else {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
      }
    }
  }

  return result;
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
    return ArrayStorage.fromDataShared(data, [size], dtype, [1], 0, storage.wasmRegion);
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
  return ArrayStorage.fromDataShared(
    data,
    newShape,
    dtype,
    newStrides,
    storage.offset,
    storage.wasmRegion
  );
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

    return ArrayStorage.fromDataShared(
      data,
      newShape,
      dtype,
      newStrides,
      storage.offset,
      storage.wasmRegion
    );
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

    return ArrayStorage.fromDataShared(
      data,
      newShape,
      dtype,
      newStrides,
      storage.offset,
      storage.wasmRegion
    );
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

  return ArrayStorage.fromDataShared(
    data,
    newShape,
    dtype,
    newStrides,
    storage.offset,
    storage.wasmRegion
  );
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
    return ArrayStorage.fromDataShared(
      data,
      Array.from(shape),
      dtype,
      Array.from(strides),
      storage.offset,
      storage.wasmRegion
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

  return ArrayStorage.fromDataShared(
    data,
    newShape,
    dtype,
    newStrides,
    storage.offset,
    storage.wasmRegion
  );
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

  // Create output array directly in WASM memory
  const result = ArrayStorage.empty(outputShape, dtype);
  const outputData = result.data;
  const outputStrides = computeStrides(outputShape);

  // Copy data from each input array
  let offset = 0;
  for (const storage of storages) {
    const axisSize = storage.shape[normalizedAxis]!;
    copyToOutput(storage, outputData, outputShape, outputStrides, normalizedAxis, offset, dtype);
    offset += axisSize;
  }

  return result;
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
  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  // Complex arrays: each logical element = 2 physical entries (re, im)
  const elemScale = isComplex ? 2 : 1;

  // Fast path: if concatenating along axis 0 and both are C-contiguous,
  // we can do bulk copy
  if (axis === 0 && source.isCContiguous && ndim > 0) {
    // Calculate the starting position in output array (physical)
    const outputOffset = axisOffset * outputStrides[0]! * elemScale;
    const sourceData = source.data;
    const start = source.offset * elemScale;
    const end = start + sourceSize * elemScale;

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
    const sourceStart = source.offset * elemScale;

    for (let row = 0; row < rows; row++) {
      const sourceRowStart = sourceStart + row * cols * elemScale;
      const outputRowStart = (row * outputCols + axisOffset) * elemScale;
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>).set(
        sourceData.subarray(sourceRowStart, sourceRowStart + cols * elemScale) as Float64Array,
        outputRowStart
      );
    }
    return;
  }

  // Float16Array optimization: bulk-convert for faster per-element access
  if (dtype === 'float16' && hasFloat16 && source.isCContiguous) {
    const f32Src = new Float32Array(
      (source.data as Float16Array).subarray(source.offset, source.offset + sourceSize)
    );
    const indices = new Array(ndim).fill(0);
    const baseOutputOffset = axisOffset * outputStrides[axis]!;

    for (let i = 0; i < sourceSize; i++) {
      let outputIdx = baseOutputOffset;
      for (let d = 0; d < ndim; d++) {
        outputIdx += indices[d]! * outputStrides[d]!;
      }
      (outputData as Float16Array)[outputIdx] = f32Src[i]!;

      for (let d = ndim - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d]! < sourceShape[d]!) break;
        indices[d] = 0;
      }
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
    if (isComplex) {
      const c = value as { re: number; im: number };
      (outputData as Float64Array)[outputIdx * 2] = c.re;
      (outputData as Float64Array)[outputIdx * 2 + 1] = c.im;
    } else if (isBigInt) {
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
  const result = concatenate(expanded, normalizedAxis);
  for (const e of expanded) e.dispose();
  return result;
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

  // WASM fast path for 2D tile
  if (ndim === 2 && repsArr.length === 2 && storage.isCContiguous) {
    const wasm = wasmTile2D(storage, repsArr[0]!, repsArr[1]!);
    if (wasm) return wasm;
  }

  // WASM fast path for 1D tile (treat as 2D [1, N] tiled by [1, reps])
  if (ndim === 1 && repsArr.length === 1 && storage.isCContiguous) {
    const wasm = wasmTile2D(storage, 1, repsArr[0]!);
    if (wasm) {
      // Reshape from [1, N*reps] to [N*reps]
      return ArrayStorage.fromData(wasm.data, [storage.size * repsArr[0]!], storage.dtype);
    }
  }

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

  const result = ArrayStorage.empty(outputShape, dtype);
  const outputData = result.data;
  const outputStrides = computeStrides(outputShape);

  // If we need to expand dimensions of input, reshape it
  let expandedStorage = storage;
  if (ndim < maxDim) {
    expandedStorage = reshape(storage, paddedShape);
  }

  const isBigInt = dtype === 'int64' || dtype === 'uint64';
  const expandedStrides = expandedStorage.strides;

  // Float16Array optimization: bulk-convert to Float32Array for faster per-element access
  if (dtype === 'float16' && hasFloat16 && expandedStorage.isCContiguous) {
    const srcSize = expandedStorage.size;
    const f32Src = new Float32Array(
      (expandedStorage.data as Float16Array).subarray(
        expandedStorage.offset,
        expandedStorage.offset + srcSize
      )
    );
    const f32Out = new Float32Array(outputSize);
    const outputIndices = new Array(maxDim).fill(0);

    for (let i = 0; i < outputSize; i++) {
      let sourceFlatIdx = 0;
      for (let d = 0; d < maxDim; d++) {
        const sourceIdx = outputIndices[d]! % paddedShape[d]!;
        sourceFlatIdx += sourceIdx * expandedStrides[d]!;
      }
      f32Out[i] = f32Src[sourceFlatIdx]!;

      for (let d = maxDim - 1; d >= 0; d--) {
        outputIndices[d]++;
        if (outputIndices[d]! < outputShape[d]!) break;
        outputIndices[d] = 0;
      }
    }

    (outputData as Float16Array).set(f32Out);
    return result;
  }

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

  return result;
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
    // WASM fast path for uniform flat repeat
    if (typeof repeats === 'number' && storage.isCContiguous) {
      const wasm = wasmRepeat(storage, repeats);
      if (wasm) return wasm;
    }

    // Flatten and repeat each element
    const flatSize = size;
    const repeatsArr = Array.isArray(repeats) ? repeats : new Array(flatSize).fill(repeats);

    if (repeatsArr.length !== flatSize) {
      throw new Error(
        `operands could not be broadcast together with shape (${flatSize},) (${repeatsArr.length},)`
      );
    }

    const outputSize = repeatsArr.reduce((a, b) => a + b, 0);
    const flatResult = ArrayStorage.empty([outputSize], dtype);
    const outputData = flatResult.data;

    // Float16Array optimization: bulk-convert for faster per-element access
    if (dtype === 'float16' && hasFloat16 && storage.isCContiguous) {
      const f32Src = new Float32Array(
        (storage.data as Float16Array).subarray(storage.offset, storage.offset + flatSize)
      );
      const f32Out = new Float32Array(outputSize);
      let outIdx = 0;
      for (let i = 0; i < flatSize; i++) {
        const value = f32Src[i]!;
        const rep = repeatsArr[i]!;
        for (let r = 0; r < rep; r++) {
          f32Out[outIdx++] = value;
        }
      }
      (outputData as Float16Array).set(f32Out);
      return flatResult;
    }

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

    return flatResult;
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

  const axisResult = ArrayStorage.empty(outputShape, dtype);
  const outputData = axisResult.data;
  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const outputStrides = computeStrides(outputShape);

  // Iterate through source and write repeated values
  const sourceIndices = new Array(ndim).fill(0);
  const isBigInt = dtype === 'int64' || dtype === 'uint64';

  // Track cumulative positions along axis
  const axisPositions: number[] = [0];
  for (let i = 0; i < axisSize; i++) {
    axisPositions.push(axisPositions[i]! + repeatsArr[i]!);
  }

  // Float16Array optimization: bulk-convert for faster per-element access
  if (dtype === 'float16' && hasFloat16 && storage.isCContiguous) {
    const f32Src = new Float32Array(
      (storage.data as Float16Array).subarray(storage.offset, storage.offset + size)
    );
    const f32Out = new Float32Array(outputSize);

    for (let i = 0; i < size; i++) {
      const value = f32Src[i]!;
      const axisIdx = sourceIndices[normalizedAxis]!;
      const rep = repeatsArr[axisIdx]!;

      let baseOutIdx = 0;
      for (let d = 0; d < ndim; d++) {
        if (d !== normalizedAxis) {
          baseOutIdx += sourceIndices[d]! * outputStrides[d]!;
        }
      }

      const axisStride = outputStrides[normalizedAxis]!;
      const axisStart = axisPositions[axisIdx]!;
      for (let r = 0; r < rep; r++) {
        f32Out[baseOutIdx + (axisStart + r) * axisStride] = value;
      }

      for (let d = ndim - 1; d >= 0; d--) {
        sourceIndices[d]++;
        if (sourceIndices[d]! < shape[d]!) break;
        sourceIndices[d] = 0;
      }
    }

    (outputData as Float16Array).set(f32Out);
    return axisResult;
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

  return axisResult;
}

/**
 * Reverse the order of elements in an array along given axis
 */
export function flip(storage: ArrayStorage, axis?: number | number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;
  // const size = storage.size;

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

  // Return a strided view — O(1), no data copied.
  // Flipping axis i: negate its stride and shift the offset so element 0
  // maps to the last element along that axis in the source.
  const strides = Array.from(storage.strides);
  let newOffset = storage.offset;
  for (const ax of axesToFlip) {
    newOffset += (shape[ax]! - 1) * strides[ax]!;
    strides[ax] = -strides[ax]!;
  }

  return ArrayStorage.fromDataShared(
    storage.data,
    [...shape],
    dtype,
    strides,
    newOffset,
    storage.wasmRegion
  );
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

  // Return a strided view — O(1), no data copied, works for any ndim and any k.
  //
  // Rotating axes (axis0, axis1) by k steps CCW is equivalent to transforming
  // the strides and offset of the view:
  //
  //   k=1: new_strides[axis0] = -s1, new_strides[axis1] = s0
  //        new_offset += (dim1-1)*s1    (dim1 = shape[axis1])
  //        swap shape[axis0] <-> shape[axis1]
  //
  //   k=2: new_strides[axis0] = -s0, new_strides[axis1] = -s1
  //        new_offset += (dim0-1)*s0 + (dim1-1)*s1
  //        shape unchanged
  //
  //   k=3: new_strides[axis0] = s1, new_strides[axis1] = -s0
  //        new_offset += (dim0-1)*s0    (dim0 = shape[axis0])
  //        swap shape[axis0] <-> shape[axis1]
  const strides = Array.from(storage.strides);
  const newShape = [...shape];
  const s0 = strides[axis0]!;
  const s1 = strides[axis1]!;
  const dim0 = shape[axis0]!;
  const dim1 = shape[axis1]!;
  let newOffset = storage.offset;

  if (k === 1) {
    strides[axis0] = -s1;
    strides[axis1] = s0;
    newOffset += (dim1 - 1) * s1;
    newShape[axis0] = dim1;
    newShape[axis1] = dim0;
  } else if (k === 2) {
    strides[axis0] = -s0;
    strides[axis1] = -s1;
    newOffset += (dim0 - 1) * s0 + (dim1 - 1) * s1;
  } else {
    // k === 3
    strides[axis0] = s1;
    strides[axis1] = -s0;
    newOffset += (dim0 - 1) * s0;
    newShape[axis0] = dim1;
    newShape[axis1] = dim0;
  }

  return ArrayStorage.fromDataShared(
    storage.data,
    newShape,
    dtype,
    strides,
    newOffset,
    storage.wasmRegion
  );
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

    // WASM fast path for flat roll
    if (storage.isCContiguous) {
      const wasm = wasmRoll(storage, flatShift);
      if (wasm) return wasm;
    }

    const flatStorage = flatten(storage);

    const rollResult = ArrayStorage.empty([...shape], dtype);
    const outputData = rollResult.data;
    const isBigInt = isBigIntDType(dtype);

    // Float16Array optimization: bulk-convert for faster per-element access
    if (dtype === 'float16' && hasFloat16 && flatStorage.isCContiguous) {
      const f32Src = new Float32Array(
        (flatStorage.data as Float16Array).subarray(flatStorage.offset, flatStorage.offset + size)
      );
      const f32Out = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const sourceIdx = (((i - flatShift) % size) + size) % size;
        f32Out[i] = f32Src[sourceIdx]!;
      }
      (outputData as Float16Array).set(f32Out);
      return rollResult;
    }

    const isComplex = isComplexDType(dtype);
    for (let i = 0; i < size; i++) {
      const sourceIdx = (((i - flatShift) % size) + size) % size;
      if (isComplex) {
        const value = flatStorage.iget(sourceIdx) as Complex;
        (outputData as Float64Array | Float32Array)[i * 2] = value.re;
        (outputData as Float64Array | Float32Array)[i * 2 + 1] = value.im;
      } else if (isBigInt) {
        (outputData as BigInt64Array | BigUint64Array)[i] = flatStorage.iget(sourceIdx) as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = flatStorage.iget(
          sourceIdx
        ) as number;
      }
    }

    // Reshape back to original shape
    return rollResult;
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
  const axisRollResult = ArrayStorage.empty([...shape], dtype);
  const outputData = axisRollResult.data;
  const isBigInt = isBigIntDType(dtype);

  // Iterate through all output positions
  const outputIndices = new Array(ndim).fill(0);

  // Float16Array optimization: bulk-convert for faster per-element access
  if (dtype === 'float16' && hasFloat16 && storage.isCContiguous) {
    const f32Src = new Float32Array(
      (storage.data as Float16Array).subarray(storage.offset, storage.offset + size)
    );
    const f32Out = new Float32Array(size);
    const srcStrides = storage.strides;

    for (let i = 0; i < size; i++) {
      const sourceIndices = [...outputIndices];
      for (let j = 0; j < normalizedAxes.length; j++) {
        const ax = normalizedAxes[j]!;
        const axisSize = shape[ax]!;
        const sh = shifts[j]!;
        sourceIndices[ax] = (((sourceIndices[ax]! - sh) % axisSize) + axisSize) % axisSize;
      }

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += sourceIndices[d]! * srcStrides[d]!;
      }
      f32Out[i] = f32Src[srcIdx]!;

      for (let d = ndim - 1; d >= 0; d--) {
        outputIndices[d]++;
        if (outputIndices[d]! < shape[d]!) break;
        outputIndices[d] = 0;
      }
    }

    (outputData as Float16Array).set(f32Out);
    return axisRollResult;
  }

  const isComplex = isComplexDType(dtype);
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

    // Write to output
    if (isComplex) {
      const physSrc = sourceOffset * 2;
      (outputData as Float64Array | Float32Array)[i * 2] = (
        storage.data as Float64Array | Float32Array
      )[physSrc]!;
      (outputData as Float64Array | Float32Array)[i * 2 + 1] = (
        storage.data as Float64Array | Float32Array
      )[physSrc + 1]!;
    } else if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[i] = storage.data[sourceOffset] as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = storage.data[
        sourceOffset
      ] as number;
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

  return axisRollResult;
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
  const resizeResult = ArrayStorage.empty(newShape, dtype);
  const outputData = resizeResult.data;
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

  return resizeResult;
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
  const results: ArrayStorage[] = new Array(axisSize);
  const data = storage.data;
  const dtype = storage.dtype;
  const strides = storage.strides;
  const baseOffset = storage.offset;

  // Build output shape and strides (axis dimension removed)
  const outShape: number[] = [];
  const outStrides: number[] = [];
  for (let d = 0; d < ndim; d++) {
    if (d !== normalizedAxis) {
      outShape.push(shape[d]!);
      outStrides.push(strides[d]!);
    }
  }

  const axisStride = strides[normalizedAxis]!;

  for (let i = 0; i < axisSize; i++) {
    results[i] = ArrayStorage.fromData(
      data,
      outShape,
      dtype,
      outStrides,
      baseOffset + i * axisStride
    );
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
