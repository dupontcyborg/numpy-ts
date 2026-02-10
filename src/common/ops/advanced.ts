/**
 * Advanced array operations
 *
 * Broadcasting, indexing, and comparison functions.
 * @module ops/advanced
 */

import { ArrayStorage, computeStrides } from '../core/storage';
import { getTypedArrayConstructor, isBigIntDType, type TypedArray } from '../core/dtype';
import { computeBroadcastShape, broadcastTo, broadcastShapes } from '../core/broadcasting';
import { Complex } from '../core/complex';

/**
 * Broadcast an array to a given shape
 * Returns a read-only view on the original array
 */
export function broadcast_to(storage: ArrayStorage, targetShape: number[]): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const targetNdim = targetShape.length;

  if (targetNdim < ndim) {
    throw new Error(`input operand has more dimensions than allowed by the axis remapping`);
  }

  // Validate that broadcasting is possible
  const broadcastedShape = computeBroadcastShape([Array.from(shape), targetShape]);
  if (broadcastedShape === null) {
    throw new Error(
      `operands could not be broadcast together with shape (${shape.join(',')}) (${targetShape.join(',')})`
    );
  }

  // Check result matches target
  for (let i = 0; i < targetNdim; i++) {
    if (broadcastedShape[i] !== targetShape[i]) {
      throw new Error(
        `operands could not be broadcast together with shape (${shape.join(',')}) (${targetShape.join(',')})`
      );
    }
  }

  return broadcastTo(storage, targetShape);
}

/**
 * Broadcast multiple arrays to a common shape
 * Returns views on the original arrays
 */
export function broadcast_arrays(storages: ArrayStorage[]): ArrayStorage[] {
  if (storages.length === 0) {
    return [];
  }

  if (storages.length === 1) {
    return [storages[0]!];
  }

  // Compute broadcast shape
  const shapes = storages.map((s) => Array.from(s.shape));
  const targetShape = computeBroadcastShape(shapes);

  if (targetShape === null) {
    throw new Error(
      `operands could not be broadcast together with shapes ${shapes.map((s) => `(${s.join(',')})`).join(' ')}`
    );
  }

  // Broadcast each array to the target shape
  return storages.map((s) => broadcastTo(s, targetShape));
}

/**
 * Compute the broadcast shape for multiple shapes
 * Re-export from core/broadcasting for convenience
 */
export { broadcastShapes as broadcast_shapes };

/**
 * Take elements from an array along an axis
 */
export function take(storage: ArrayStorage, indices: number[], axis?: number): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  if (axis === undefined) {
    // Flatten and take
    const flatSize = storage.size;

    // Validate indices
    for (const idx of indices) {
      const normalizedIdx = idx < 0 ? flatSize + idx : idx;
      if (normalizedIdx < 0 || normalizedIdx >= flatSize) {
        throw new Error(`index ${idx} is out of bounds for axis 0 with size ${flatSize}`);
      }
    }

    // Create output array
    const outputSize = indices.length;
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot take from array with dtype ${dtype}`);
    }
    const outputData = new Constructor(outputSize);

    for (let i = 0; i < outputSize; i++) {
      let idx = indices[i]!;
      if (idx < 0) idx = flatSize + idx;
      const value = storage.iget(idx);

      if (isBigIntDType(dtype)) {
        (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
      } else {
        (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
      }
    }

    return ArrayStorage.fromData(outputData, [outputSize], dtype);
  }

  // Take along specified axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;

  // Validate indices
  for (const idx of indices) {
    const normalizedIdx = idx < 0 ? axisSize + idx : idx;
    if (normalizedIdx < 0 || normalizedIdx >= axisSize) {
      throw new Error(
        `index ${idx} is out of bounds for axis ${normalizedAxis} with size ${axisSize}`
      );
    }
  }

  // Calculate output shape
  const outputShape = Array.from(shape);
  outputShape[normalizedAxis] = indices.length;

  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot take from array with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const outputStrides = computeStrides(outputShape);

  // Iterate through output positions
  const outputIndices = new Array(ndim).fill(0);
  for (let i = 0; i < outputSize; i++) {
    // Compute source index
    const sourceIndices = [...outputIndices];
    let targetIdx = outputIndices[normalizedAxis]!;
    let sourceAxisIdx = indices[targetIdx]!;
    if (sourceAxisIdx < 0) sourceAxisIdx = axisSize + sourceAxisIdx;
    sourceIndices[normalizedAxis] = sourceAxisIdx;

    const value = storage.get(...sourceIndices);

    // Write to output
    let outIdx = 0;
    for (let d = 0; d < ndim; d++) {
      outIdx += outputIndices[d]! * outputStrides[d]!;
    }

    if (isBigIntDType(dtype)) {
      (outputData as BigInt64Array | BigUint64Array)[outIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx] = value as number;
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
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
 * Put values at specified indices (modifies array in-place)
 */
export function put(
  storage: ArrayStorage,
  indices: number[],
  values: ArrayStorage | number | bigint
): void {
  const flatSize = storage.size;
  const dtype = storage.dtype;

  // Get values to put
  let valueArray: (number | bigint)[];
  if (typeof values === 'number' || typeof values === 'bigint') {
    valueArray = new Array(indices.length).fill(values);
  } else {
    // Extract values from storage
    valueArray = [];
    for (let i = 0; i < values.size; i++) {
      const val = values.iget(i);
      // Convert Complex to its real part for non-complex operations
      valueArray.push(val instanceof Complex ? val.re : (val as number | bigint));
    }
    // Broadcast values if needed
    if (valueArray.length === 1) {
      valueArray = new Array(indices.length).fill(valueArray[0]);
    } else if (valueArray.length !== indices.length) {
      // Tile values to match indices length
      const original = [...valueArray];
      valueArray = [];
      for (let i = 0; i < indices.length; i++) {
        valueArray.push(original[i % original.length]!);
      }
    }
  }

  // Put values at indices
  for (let i = 0; i < indices.length; i++) {
    let idx = indices[i]!;
    if (idx < 0) idx = flatSize + idx;

    if (idx < 0 || idx >= flatSize) {
      throw new Error(`index ${indices[i]} is out of bounds for axis 0 with size ${flatSize}`);
    }

    let value = valueArray[i]!;

    // Convert value to appropriate type
    if (isBigIntDType(dtype)) {
      if (typeof value !== 'bigint') {
        value = BigInt(Math.round(Number(value)));
      }
    } else {
      if (typeof value === 'bigint') {
        value = Number(value);
      }
    }

    storage.iset(idx, value);
  }
}

/**
 * Construct array from index array and choices
 */
export function choose(indexStorage: ArrayStorage, choices: ArrayStorage[]): ArrayStorage {
  if (choices.length === 0) {
    throw new Error('choices cannot be empty');
  }

  const indexShape = indexStorage.shape;
  const numChoices = choices.length;
  const dtype = choices[0]!.dtype;

  // Validate that all choices have compatible shapes
  const shapes = choices.map((c) => Array.from(c.shape));
  shapes.unshift(Array.from(indexShape));
  const broadcastedShape = computeBroadcastShape(shapes);

  if (broadcastedShape === null) {
    throw new Error('operands could not be broadcast together');
  }

  // Broadcast index array and choices to common shape
  const broadcastedIndex = broadcastTo(indexStorage, broadcastedShape);
  const broadcastedChoices = choices.map((c) => broadcastTo(c, broadcastedShape));

  // Create output array
  const outputSize = broadcastedShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot choose with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);

  // Fill output
  for (let i = 0; i < outputSize; i++) {
    const choiceIdx = Number(broadcastedIndex.iget(i));

    if (choiceIdx < 0 || choiceIdx >= numChoices) {
      throw new Error(`index ${choiceIdx} is out of bounds for axis 0 with size ${numChoices}`);
    }

    const value = broadcastedChoices[choiceIdx]!.iget(i);

    if (isBigIntDType(dtype)) {
      (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
    }
  }

  return ArrayStorage.fromData(outputData, broadcastedShape, dtype);
}

/**
 * Check if two arrays are element-wise equal
 */
export function array_equal(a: ArrayStorage, b: ArrayStorage, equal_nan: boolean = false): boolean {
  // Check shapes match
  if (a.ndim !== b.ndim) {
    return false;
  }

  for (let i = 0; i < a.ndim; i++) {
    if (a.shape[i] !== b.shape[i]) {
      return false;
    }
  }

  // Check all elements
  const size = a.size;
  for (let i = 0; i < size; i++) {
    const aVal = a.iget(i);
    const bVal = b.iget(i);

    // Handle NaN comparison
    if (equal_nan) {
      const aIsNaN = typeof aVal === 'number' && Number.isNaN(aVal);
      const bIsNaN = typeof bVal === 'number' && Number.isNaN(bVal);
      if (aIsNaN && bIsNaN) {
        continue;
      }
    }

    if (aVal !== bVal) {
      return false;
    }
  }

  return true;
}

/**
 * Take values along an axis using 1D index array
 */
export function take_along_axis(
  storage: ArrayStorage,
  indices: ArrayStorage,
  axis: number
): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // indices must have same ndim as storage
  const indicesShape = indices.shape;
  if (indicesShape.length !== ndim) {
    throw new Error(
      `indices and arr must have the same number of dimensions, got ${indicesShape.length} vs ${ndim}`
    );
  }

  // Check that non-axis dimensions match (or are broadcastable with 1)
  for (let i = 0; i < ndim; i++) {
    if (i !== normalizedAxis) {
      if (indicesShape[i] !== shape[i] && indicesShape[i] !== 1 && shape[i] !== 1) {
        throw new Error(
          `index ${indicesShape[i]} is out of bounds for size ${shape[i]} in dimension ${i}`
        );
      }
    }
  }

  // Output shape matches indices shape
  const outputShape = Array.from(indicesShape);
  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot take_along_axis with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);
  const inputStrides = computeStrides(shape);
  const indicesStrides = computeStrides(indicesShape);

  const axisSize = shape[normalizedAxis]!;

  // Iterate through output positions
  for (let outIdx = 0; outIdx < outputSize; outIdx++) {
    // Convert outIdx to multi-index in output shape
    const multiIdx = new Array(ndim);
    let remaining = outIdx;
    for (let d = ndim - 1; d >= 0; d--) {
      multiIdx[d] = remaining % outputShape[d]!;
      remaining = Math.floor(remaining / outputShape[d]!);
    }

    // Get the index value from indices array
    let indicesLinearIdx = 0;
    for (let d = 0; d < ndim; d++) {
      const idx = indicesShape[d] === 1 ? 0 : multiIdx[d]!;
      indicesLinearIdx += idx * indicesStrides[d]!;
    }
    let indexValue = Number(indices.iget(indicesLinearIdx));
    if (indexValue < 0) indexValue = axisSize + indexValue;
    if (indexValue < 0 || indexValue >= axisSize) {
      throw new Error(
        `index ${indexValue} is out of bounds for axis ${normalizedAxis} with size ${axisSize}`
      );
    }

    // Compute source index
    const sourceMultiIdx = [...multiIdx];
    sourceMultiIdx[normalizedAxis] = indexValue;
    let srcLinearIdx = 0;
    for (let d = 0; d < ndim; d++) {
      const idx = shape[d] === 1 ? 0 : sourceMultiIdx[d]!;
      srcLinearIdx += idx * inputStrides[d]!;
    }

    const value = storage.iget(srcLinearIdx);
    if (isBigIntDType(dtype)) {
      (outputData as BigInt64Array | BigUint64Array)[outIdx] = value as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx] = value as number;
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Put values into array along an axis using 1D index array
 */
export function put_along_axis(
  storage: ArrayStorage,
  indices: ArrayStorage,
  values: ArrayStorage,
  axis: number
): void {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  // Normalize axis
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const indicesShape = indices.shape;
  const valuesShape = values.shape;

  if (indicesShape.length !== ndim || valuesShape.length !== ndim) {
    throw new Error('indices, arr, and values must have same ndim');
  }

  const axisSize = shape[normalizedAxis]!;
  const inputStrides = computeStrides(shape);
  const indicesStrides = computeStrides(indicesShape);
  const valuesStrides = computeStrides(valuesShape);

  // Iterate through indices positions
  const indicesSize = indicesShape.reduce((a, b) => a * b, 1);
  for (let idx = 0; idx < indicesSize; idx++) {
    // Convert idx to multi-index
    const multiIdx = new Array(ndim);
    let remaining = idx;
    for (let d = ndim - 1; d >= 0; d--) {
      multiIdx[d] = remaining % indicesShape[d]!;
      remaining = Math.floor(remaining / indicesShape[d]!);
    }

    // Get the index value
    let indicesLinearIdx = 0;
    for (let d = 0; d < ndim; d++) {
      indicesLinearIdx += multiIdx[d]! * indicesStrides[d]!;
    }
    let indexValue = Number(indices.iget(indicesLinearIdx));
    if (indexValue < 0) indexValue = axisSize + indexValue;
    if (indexValue < 0 || indexValue >= axisSize) {
      throw new Error(
        `index ${indexValue} is out of bounds for axis ${normalizedAxis} with size ${axisSize}`
      );
    }

    // Get value from values array (broadcast if needed)
    let valuesLinearIdx = 0;
    for (let d = 0; d < ndim; d++) {
      const vidx = valuesShape[d] === 1 ? 0 : multiIdx[d]!;
      valuesLinearIdx += vidx * valuesStrides[d]!;
    }
    let value = values.iget(valuesLinearIdx);

    // Compute destination index
    const destMultiIdx = [...multiIdx];
    destMultiIdx[normalizedAxis] = indexValue;
    let destLinearIdx = 0;
    for (let d = 0; d < ndim; d++) {
      destLinearIdx += destMultiIdx[d]! * inputStrides[d]!;
    }

    // Convert type if needed
    if (isBigIntDType(dtype)) {
      if (typeof value !== 'bigint') {
        value = BigInt(Math.round(Number(value)));
      }
    } else {
      if (typeof value === 'bigint') {
        value = Number(value);
      }
    }

    storage.iset(destLinearIdx, value);
  }
}

/**
 * Change elements of array based on conditional mask
 */
export function putmask(
  storage: ArrayStorage,
  mask: ArrayStorage,
  values: ArrayStorage | number | bigint
): void {
  const size = storage.size;
  const dtype = storage.dtype;

  // Get values array
  let valueArray: (number | bigint)[];
  if (typeof values === 'number' || typeof values === 'bigint') {
    valueArray = [values];
  } else {
    valueArray = [];
    for (let i = 0; i < values.size; i++) {
      const val = values.iget(i);
      // Convert Complex to its real part for non-complex operations
      valueArray.push(val instanceof Complex ? val.re : (val as number | bigint));
    }
  }

  // Put values where mask is true
  let valueIdx = 0;
  for (let i = 0; i < size; i++) {
    const maskVal = mask.iget(i);
    if (maskVal) {
      let value = valueArray[valueIdx % valueArray.length]!;

      // Convert type if needed
      if (isBigIntDType(dtype)) {
        if (typeof value !== 'bigint') {
          value = BigInt(Math.round(Number(value)));
        }
      } else {
        if (typeof value === 'bigint') {
          value = Number(value);
        }
      }

      storage.iset(i, value);
      valueIdx++;
    }
  }
}

/**
 * Return selected slices along given axis based on condition
 */
export function compress(
  condition: ArrayStorage,
  storage: ArrayStorage,
  axis?: number
): ArrayStorage {
  const shape = storage.shape;
  const ndim = shape.length;
  const dtype = storage.dtype;

  // Get direct access to underlying data for fast reading
  const inputData = storage.data;
  const isBigInt = isBigIntDType(dtype);

  if (axis === undefined) {
    // Flatten and select - optimized path
    // First pass: count true values
    let trueCount = 0;
    const maxLen = Math.min(condition.size, storage.size);
    for (let i = 0; i < maxLen; i++) {
      if (condition.iget(i)) trueCount++;
    }

    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot compress with dtype ${dtype}`);
    }
    const outputData = new Constructor(trueCount);

    // Second pass: copy values
    let outIdx = 0;
    for (let i = 0; i < maxLen; i++) {
      if (condition.iget(i)) {
        if (isBigInt) {
          (outputData as BigInt64Array | BigUint64Array)[outIdx] = (
            inputData as BigInt64Array | BigUint64Array
          )[i]!;
        } else {
          (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[outIdx] = (
            inputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>
          )[i]!;
        }
        outIdx++;
      }
    }

    return ArrayStorage.fromData(outputData, [trueCount], dtype);
  }

  // Compress along axis - optimized version
  const normalizedAxis = axis < 0 ? ndim + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Build boolean array and axis mapping in one pass
  const axisSize = shape[normalizedAxis]!;
  const maxLen = Math.min(condition.size, axisSize);
  const axisMap: number[] = [];

  for (let i = 0; i < maxLen; i++) {
    if (condition.iget(i)) {
      axisMap.push(i);
    }
  }

  const trueCount = axisMap.length;

  // Output shape
  const outputShape = [...shape];
  outputShape[normalizedAxis] = trueCount;
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot compress with dtype ${dtype}`);
  }
  const outputData = new Constructor(outputSize);

  // Compute strides for efficient indexing
  const inputStrides = computeStrides(shape);

  // Special case: axis = 0 (most common, optimize heavily)
  if (normalizedAxis === 0) {
    const strideAlongAxis = inputStrides[0]!;
    const elementsPerSlice = shape.slice(1).reduce((a, b) => a * b, 1);

    let outIdx = 0;
    for (let i = 0; i < trueCount; i++) {
      const inputAxisIdx = axisMap[i]!;
      const srcOffset = inputAxisIdx * strideAlongAxis;

      // Copy entire slice at once
      if (isBigInt) {
        const src = inputData as BigInt64Array | BigUint64Array;
        const dst = outputData as BigInt64Array | BigUint64Array;
        for (let j = 0; j < elementsPerSlice; j++) {
          dst[outIdx++] = src[srcOffset + j]!;
        }
      } else {
        const src = inputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        const dst = outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
        for (let j = 0; j < elementsPerSlice; j++) {
          dst[outIdx++] = src[srcOffset + j]!;
        }
      }
    }
  } else {
    // General case for other axes
    // Pre-compute outer and inner iteration counts
    const outerSize = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);

    let outIdx = 0;
    for (let outer = 0; outer < outerSize; outer++) {
      for (let axisIdx = 0; axisIdx < trueCount; axisIdx++) {
        const inputAxisIdx = axisMap[axisIdx]!;

        // Compute base offset for this outer/axis combination
        let baseOffset = 0;
        let rem = outer;
        for (let d = normalizedAxis - 1; d >= 0; d--) {
          const idx = rem % shape[d]!;
          rem = Math.floor(rem / shape[d]!);
          baseOffset += idx * inputStrides[d]!;
        }
        baseOffset += inputAxisIdx * inputStrides[normalizedAxis]!;

        // Copy inner elements
        if (isBigInt) {
          const src = inputData as BigInt64Array | BigUint64Array;
          const dst = outputData as BigInt64Array | BigUint64Array;
          for (let inner = 0; inner < innerSize; inner++) {
            dst[outIdx++] = src[baseOffset + inner]!;
          }
        } else {
          const src = inputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
          const dst = outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
          for (let inner = 0; inner < innerSize; inner++) {
            dst[outIdx++] = src[baseOffset + inner]!;
          }
        }
      }
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Return array drawn from elements in choicelist, depending on conditions
 */
export function select(
  condlist: ArrayStorage[],
  choicelist: ArrayStorage[],
  defaultValue: number | bigint = 0
): ArrayStorage {
  if (condlist.length !== choicelist.length) {
    throw new Error('condlist and choicelist must have same length');
  }

  if (condlist.length === 0) {
    throw new Error('condlist and choicelist cannot be empty');
  }

  // Compute broadcast shape from all conditions and choices
  const allShapes = [
    ...condlist.map((c) => Array.from(c.shape)),
    ...choicelist.map((c) => Array.from(c.shape)),
  ];
  const outputShape = computeBroadcastShape(allShapes);
  if (outputShape === null) {
    throw new Error('condlist and choicelist arrays could not be broadcast together');
  }

  const dtype = choicelist[0]!.dtype;
  const outputSize = outputShape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot select with dtype ${dtype}`);
  }

  // Initialize with default value
  let defaultVal: number | bigint = defaultValue;
  if (isBigIntDType(dtype)) {
    defaultVal = typeof defaultValue === 'bigint' ? defaultValue : BigInt(defaultValue);
  } else {
    defaultVal = typeof defaultValue === 'bigint' ? Number(defaultValue) : defaultValue;
  }

  const outputData = new Constructor(outputSize);
  for (let i = 0; i < outputSize; i++) {
    if (isBigIntDType(dtype)) {
      (outputData as BigInt64Array | BigUint64Array)[i] = defaultVal as bigint;
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = defaultVal as number;
    }
  }

  // Broadcast all arrays
  const broadcastedConds = condlist.map((c) => broadcastTo(c, outputShape));
  const broadcastedChoices = choicelist.map((c) => broadcastTo(c, outputShape));

  // Process conditions in order (first match wins)
  for (let i = 0; i < outputSize; i++) {
    for (let j = 0; j < condlist.length; j++) {
      if (broadcastedConds[j]!.iget(i)) {
        const value = broadcastedChoices[j]!.iget(i);
        if (isBigIntDType(dtype)) {
          (outputData as BigInt64Array | BigUint64Array)[i] = value as bigint;
        } else {
          (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value as number;
        }
        break;
      }
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Change elements of array based on conditional and input values
 */
export function place(storage: ArrayStorage, mask: ArrayStorage, vals: ArrayStorage): void {
  const size = storage.size;
  const dtype = storage.dtype;

  // Get values array
  const valueArray: (number | bigint)[] = [];
  for (let i = 0; i < vals.size; i++) {
    const val = vals.iget(i);
    // Convert Complex to its real part for non-complex operations
    valueArray.push(val instanceof Complex ? val.re : (val as number | bigint));
  }

  if (valueArray.length === 0) {
    return;
  }

  // Place values where mask is true
  let valueIdx = 0;
  for (let i = 0; i < size; i++) {
    const maskVal = mask.iget(i);
    if (maskVal) {
      let value = valueArray[valueIdx % valueArray.length]!;

      // Convert type if needed
      if (isBigIntDType(dtype)) {
        if (typeof value !== 'bigint') {
          value = BigInt(Math.round(Number(value)));
        }
      } else {
        if (typeof value === 'bigint') {
          value = Number(value);
        }
      }

      storage.iset(i, value);
      valueIdx++;
    }
  }
}

/**
 * Return indices to access main diagonal of array
 */
export function diag_indices(n: number, ndim: number = 2): ArrayStorage[] {
  if (ndim < 1) {
    throw new Error('ndim must be at least 1');
  }

  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    indices[i] = i;
  }

  const result: ArrayStorage[] = [];
  for (let d = 0; d < ndim; d++) {
    result.push(ArrayStorage.fromData(new Int32Array(indices), [n], 'int32'));
  }

  return result;
}

/**
 * Return indices to access main diagonal of array from given array
 */
export function diag_indices_from(storage: ArrayStorage): ArrayStorage[] {
  const shape = storage.shape;
  const ndim = shape.length;

  if (ndim < 2) {
    throw new Error('array must be at least 2-D');
  }

  // Check that all dimensions are equal
  const n = shape[0]!;
  for (let i = 1; i < ndim; i++) {
    if (shape[i] !== n) {
      throw new Error('All dimensions of input must be equal');
    }
  }

  return diag_indices(n, ndim);
}

/**
 * Return indices for lower-triangle of an (n, m) array
 */
export function tril_indices(n: number, k: number = 0, m?: number): ArrayStorage[] {
  const cols = m ?? n;

  const rows: number[] = [];
  const colIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= Math.min(i + k, cols - 1); j++) {
      if (j >= 0) {
        rows.push(i);
        colIndices.push(j);
      }
    }
  }

  return [
    ArrayStorage.fromData(new Int32Array(rows), [rows.length], 'int32'),
    ArrayStorage.fromData(new Int32Array(colIndices), [colIndices.length], 'int32'),
  ];
}

/**
 * Return indices for lower-triangle of given array
 */
export function tril_indices_from(storage: ArrayStorage, k: number = 0): ArrayStorage[] {
  const shape = storage.shape;

  if (shape.length !== 2) {
    throw new Error('array must be 2-D');
  }

  return tril_indices(shape[0]!, k, shape[1]);
}

/**
 * Return indices for upper-triangle of an (n, m) array
 */
export function triu_indices(n: number, k: number = 0, m?: number): ArrayStorage[] {
  const cols = m ?? n;

  const rows: number[] = [];
  const colIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = Math.max(i + k, 0); j < cols; j++) {
      rows.push(i);
      colIndices.push(j);
    }
  }

  return [
    ArrayStorage.fromData(new Int32Array(rows), [rows.length], 'int32'),
    ArrayStorage.fromData(new Int32Array(colIndices), [colIndices.length], 'int32'),
  ];
}

/**
 * Return indices for upper-triangle of given array
 */
export function triu_indices_from(storage: ArrayStorage, k: number = 0): ArrayStorage[] {
  const shape = storage.shape;

  if (shape.length !== 2) {
    throw new Error('array must be 2-D');
  }

  return triu_indices(shape[0]!, k, shape[1]);
}

/**
 * Return indices to access elements using mask function
 */
export function mask_indices(
  n: number,
  mask_func: (n: number, k: number) => ArrayStorage,
  k: number = 0
): ArrayStorage[] {
  // Generate the mask using the mask function
  const mask = mask_func(n, k);
  const maskShape = mask.shape;

  if (maskShape.length !== 2 || maskShape[0] !== n || maskShape[1] !== n) {
    throw new Error('mask_func must return n x n array');
  }

  const rows: number[] = [];
  const cols: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (mask.get(i, j)) {
        rows.push(i);
        cols.push(j);
      }
    }
  }

  return [
    ArrayStorage.fromData(new Int32Array(rows), [rows.length], 'int32'),
    ArrayStorage.fromData(new Int32Array(cols), [cols.length], 'int32'),
  ];
}

/**
 * Return array representing indices of a grid
 */
export function indices(
  dimensions: number[],
  dtype: 'int32' | 'int64' | 'float64' = 'int32'
): ArrayStorage {
  const ndim = dimensions.length;
  const outputShape = [ndim, ...dimensions];
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create indices with dtype ${dtype}`);
  }

  const outputData = new Constructor(outputSize);
  const gridSize = dimensions.reduce((a, b) => a * b, 1);

  // For each dimension, fill the corresponding slice
  for (let d = 0; d < ndim; d++) {
    const sliceOffset = d * gridSize;

    // Iterate through grid positions
    for (let gridIdx = 0; gridIdx < gridSize; gridIdx++) {
      // Convert gridIdx to multi-index
      const multiIdx = new Array(ndim);
      let remaining = gridIdx;
      for (let i = ndim - 1; i >= 0; i--) {
        multiIdx[i] = remaining % dimensions[i]!;
        remaining = Math.floor(remaining / dimensions[i]!);
      }

      const value = multiIdx[d]!;
      if (dtype === 'int64') {
        (outputData as BigInt64Array)[sliceOffset + gridIdx] = BigInt(value);
      } else {
        (outputData as Float64Array | Int32Array)[sliceOffset + gridIdx] = value;
      }
    }
  }

  return ArrayStorage.fromData(outputData, outputShape, dtype);
}

/**
 * Construct open mesh from multiple sequences
 */
export function ix_(...args: ArrayStorage[]): ArrayStorage[] {
  const ndim = args.length;
  const result: ArrayStorage[] = [];

  for (let i = 0; i < ndim; i++) {
    const arr = args[i]!;
    const arrSize = arr.size;
    const dtype = arr.dtype;

    // Create shape with 1s except at position i
    const shape = new Array(ndim).fill(1);
    shape[i] = arrSize;

    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot create ix_ with dtype ${dtype}`);
    }

    const data = new Constructor(arrSize);
    for (let j = 0; j < arrSize; j++) {
      const value = arr.iget(j);
      if (isBigIntDType(dtype)) {
        (data as BigInt64Array | BigUint64Array)[j] = value as bigint;
      } else {
        (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[j] = value as number;
      }
    }

    result.push(ArrayStorage.fromData(data, shape, dtype));
  }

  return result;
}

/**
 * Convert multi-dimensional index arrays to flat index array
 */
export function ravel_multi_index(
  multi_index: ArrayStorage[],
  dims: number[],
  mode: 'raise' | 'wrap' | 'clip' = 'raise'
): ArrayStorage {
  if (multi_index.length !== dims.length) {
    throw new Error('multi_index length must equal dims length');
  }

  if (multi_index.length === 0) {
    throw new Error('multi_index cannot be empty');
  }

  const size = multi_index[0]!.size;
  const ndim = dims.length;
  const outputData = new Int32Array(size);

  // Compute strides for row-major (C) order
  const strides = new Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= dims[i]!;
  }

  for (let i = 0; i < size; i++) {
    let flatIdx = 0;
    for (let d = 0; d < ndim; d++) {
      let idx = Number(multi_index[d]!.iget(i));
      const dimSize = dims[d]!;

      // Handle mode
      if (mode === 'wrap') {
        idx = ((idx % dimSize) + dimSize) % dimSize;
      } else if (mode === 'clip') {
        idx = Math.max(0, Math.min(idx, dimSize - 1));
      } else if (idx < 0 || idx >= dimSize) {
        throw new Error(`index ${idx} is out of bounds for axis ${d} with size ${dimSize}`);
      }

      flatIdx += idx * strides[d]!;
    }
    outputData[i] = flatIdx;
  }

  return ArrayStorage.fromData(outputData, [size], 'int32');
}

/**
 * Convert flat index array to tuple of coordinate arrays
 */
export function unravel_index(
  indices: ArrayStorage | number,
  shape: number[],
  order: 'C' | 'F' = 'C'
): ArrayStorage[] {
  const ndim = shape.length;

  // Handle scalar input
  let indicesArray: number[];
  let outputShape: number[];
  if (typeof indices === 'number') {
    indicesArray = [indices];
    outputShape = [];
  } else {
    indicesArray = [];
    for (let i = 0; i < indices.size; i++) {
      indicesArray.push(Number(indices.iget(i)));
    }
    outputShape = Array.from(indices.shape);
  }

  const size = indicesArray.length;
  const totalSize = shape.reduce((a, b) => a * b, 1);

  // Compute strides
  const strides = new Array(ndim);
  if (order === 'C') {
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i]!;
    }
  } else {
    let stride = 1;
    for (let i = 0; i < ndim; i++) {
      strides[i] = stride;
      stride *= shape[i]!;
    }
  }

  // Create output arrays
  const result: ArrayStorage[] = [];
  for (let d = 0; d < ndim; d++) {
    const data = new Int32Array(size);
    result.push(ArrayStorage.fromData(data, outputShape.length ? outputShape : [1], 'int32'));
  }

  // Convert each flat index
  for (let i = 0; i < size; i++) {
    let flatIdx = indicesArray[i]!;
    if (flatIdx < 0 || flatIdx >= totalSize) {
      throw new Error(`index ${flatIdx} is out of bounds for array with size ${totalSize}`);
    }

    if (order === 'C') {
      for (let d = 0; d < ndim; d++) {
        const coord = Math.floor(flatIdx / strides[d]!);
        flatIdx = flatIdx % strides[d]!;
        (result[d]!.data as Int32Array)[i] = coord % shape[d]!;
      }
    } else {
      for (let d = ndim - 1; d >= 0; d--) {
        const coord = Math.floor(flatIdx / strides[d]!);
        flatIdx = flatIdx % strides[d]!;
        (result[d]!.data as Int32Array)[i] = coord % shape[d]!;
      }
    }
  }

  // For scalar input, return scalar-shaped results
  if (typeof indices === 'number') {
    return result.map((arr) => {
      const value = arr.iget(0);
      return ArrayStorage.fromData(new Int32Array([Number(value)]), [], 'int32');
    });
  }

  return result;
}

/**
 * Fill the main diagonal of a given array (modifies in-place)
 * @param a - Array storage (at least 2D)
 * @param val - Value or array of values to fill diagonal with
 * @param wrap - Whether to wrap for tall matrices
 */
export function fill_diagonal(
  a: ArrayStorage,
  val: ArrayStorage | number,
  wrap: boolean = false
): void {
  const shape = a.shape;
  const ndim = shape.length;

  if (ndim < 2) {
    throw new Error('array must be at least 2-d');
  }

  // Determine the step size to move along diagonal
  // For a 2D array [m, n], diagonal step is n + 1
  // For higher dims, it's more complex
  let step: number;
  if (ndim === 2) {
    step = shape[1]! + 1;
  } else {
    // For ND arrays, step is sum of products of remaining dimensions
    step = 1;
    for (let i = 1; i < ndim; i++) {
      let prod = 1;
      for (let j = i; j < ndim; j++) {
        prod *= shape[j]!;
      }
      step += prod;
    }
  }

  const data = a.data;
  const size = a.size;

  // Determine diagonal length
  let diagLength = Math.min(...shape);
  if (wrap && ndim === 2) {
    // For tall matrices with wrap=True, diagonal can wrap around
    diagLength = Math.max(shape[0]!, shape[1]!);
  }

  if (typeof val === 'number') {
    // Fill with scalar
    for (let i = 0; i < diagLength && i * step < size; i++) {
      const idx = i * step;
      if (idx < size) {
        data[idx] = val;
      } else {
        break;
      }
    }
  } else {
    // Fill with array values
    const valData = val.data;
    const valSize = val.size;
    for (let i = 0; i < diagLength && i * step < size; i++) {
      const idx = i * step;
      if (idx < size) {
        data[idx] = valData[i % valSize]!;
      } else {
        break;
      }
    }
  }
}

// ========================================
// Utility Functions
// ========================================

/**
 * Apply a function along a given axis.
 *
 * @param arr - Input array storage
 * @param axis - Axis along which to apply the function
 * @param func1d - Function that takes a 1D array and returns a 1D array or scalar
 * @returns Result array
 */
export function apply_along_axis(
  arr: ArrayStorage,
  axis: number,
  func1d: (slice: ArrayStorage) => ArrayStorage | number
): ArrayStorage {
  const shape = Array.from(arr.shape);
  const ndim = shape.length;

  // Normalize axis
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Calculate iteration shape (all dimensions except the axis dimension)
  const iterShape: number[] = [];
  for (let i = 0; i < ndim; i++) {
    if (i !== axis) {
      iterShape.push(shape[i]!);
    }
  }

  // If no iteration dimensions, just apply the function to the whole array
  if (iterShape.length === 0) {
    const result = func1d(arr);
    if (typeof result === 'number') {
      const resultArr = ArrayStorage.zeros([1], arr.dtype);
      resultArr.data[0] = result;
      return resultArr;
    }
    return result;
  }

  // For simplicity, we'll handle the 2D case directly
  if (ndim === 2) {
    const [rows, cols] = shape;

    if (axis === 0) {
      // Apply function to each column
      const results: (ArrayStorage | number)[] = [];
      for (let c = 0; c < cols!; c++) {
        // Extract column as 1D array
        const colData = new Float64Array(rows!);
        for (let r = 0; r < rows!; r++) {
          colData[r] = Number(arr.data[r * cols! + c]!);
        }
        const colArr = ArrayStorage.fromData(colData, [rows!], 'float64');
        results.push(func1d(colArr));
      }

      // Determine output shape and build result
      const firstResult = results[0];
      if (firstResult === undefined) {
        return ArrayStorage.zeros([0], 'float64');
      }
      if (typeof firstResult === 'number') {
        // Scalar output for each slice
        const resultArr = ArrayStorage.zeros([cols!], 'float64');
        for (let c = 0; c < cols!; c++) {
          resultArr.data[c] = results[c] as number;
        }
        return resultArr;
      } else {
        // Array output for each slice
        const resultShape = [firstResult.size, cols!];
        const resultArr = ArrayStorage.zeros(resultShape, 'float64');
        for (let c = 0; c < cols!; c++) {
          const res = results[c] as ArrayStorage;
          for (let r = 0; r < res.size; r++) {
            resultArr.data[r * cols! + c] = Number(res.data[r]!);
          }
        }
        return resultArr;
      }
    } else {
      // Apply function to each row
      const results: (ArrayStorage | number)[] = [];
      for (let r = 0; r < rows!; r++) {
        // Extract row as 1D array
        const rowData = new Float64Array(cols!);
        for (let c = 0; c < cols!; c++) {
          rowData[c] = Number(arr.data[r * cols! + c]!);
        }
        const rowArr = ArrayStorage.fromData(rowData, [cols!], 'float64');
        results.push(func1d(rowArr));
      }

      // Determine output shape and build result
      const firstResult = results[0];
      if (firstResult === undefined) {
        return ArrayStorage.zeros([0], 'float64');
      }
      if (typeof firstResult === 'number') {
        // Scalar output for each slice
        const resultArr = ArrayStorage.zeros([rows!], 'float64');
        for (let r = 0; r < rows!; r++) {
          resultArr.data[r] = results[r] as number;
        }
        return resultArr;
      } else {
        // Array output for each slice
        const resultShape = [rows!, firstResult.size];
        const resultArr = ArrayStorage.zeros(resultShape, 'float64');
        for (let r = 0; r < rows!; r++) {
          const res = results[r] as ArrayStorage;
          for (let c = 0; c < res.size; c++) {
            resultArr.data[r * res.size + c] = Number(res.data[c]!);
          }
        }
        return resultArr;
      }
    }
  }

  // For 1D arrays
  if (ndim === 1) {
    const result = func1d(arr);
    if (typeof result === 'number') {
      const resultArr = ArrayStorage.zeros([1], 'float64');
      resultArr.data[0] = result;
      return resultArr;
    }
    return result;
  }

  // General case: simplified handling
  // This is a basic implementation for common cases
  throw new Error(
    `apply_along_axis not fully implemented for ${ndim}D arrays. Only 1D and 2D arrays are supported.`
  );
}

/**
 * Apply a function over multiple axes.
 *
 * @param arr - Input array storage
 * @param func - Function that operates on an array
 * @param axes - Axes over which to apply the function
 * @returns Result array
 */
export function apply_over_axes(
  arr: ArrayStorage,
  func: (a: ArrayStorage, axis: number) => ArrayStorage,
  axes: number[]
): ArrayStorage {
  let result = arr;
  const ndim = arr.shape.length;

  for (const axis of axes) {
    // Normalize axis
    let normalizedAxis = axis < 0 ? axis + ndim : axis;
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Apply function along axis and keep dimensions
    result = func(result, normalizedAxis);

    // NumPy's apply_over_axes keeps dimensions if the result has fewer dimensions
    // by inserting a new axis
    if (result.shape.length < ndim) {
      const newShape = Array.from(result.shape);
      newShape.splice(normalizedAxis, 0, 1);
      const newStrides = computeStrides(newShape);
      result = new ArrayStorage(result.data, newShape, newStrides, 0, result.dtype);
    }
  }

  return result;
}

/**
 * Check if two arrays may share memory.
 *
 * In JavaScript, we can't directly check memory sharing like in Python.
 * This is a conservative implementation that returns true if they share
 * the same underlying buffer.
 *
 * @param a - First array storage
 * @param b - Second array storage
 * @returns True if arrays may share memory
 */
export function may_share_memory(a: ArrayStorage, b: ArrayStorage): boolean {
  // In JavaScript, we can check if the underlying typed arrays share the same buffer
  return a.data.buffer === b.data.buffer;
}

/**
 * Check if two arrays share memory.
 *
 * This is the same as may_share_memory in our implementation since
 * JavaScript doesn't have the same memory model as Python/NumPy.
 *
 * @param a - First array storage
 * @param b - Second array storage
 * @returns True if arrays share memory
 */
export function shares_memory(a: ArrayStorage, b: ArrayStorage): boolean {
  return may_share_memory(a, b);
}

// Global floating-point error state
let _floatErrorState = {
  divide: 'warn' as 'ignore' | 'warn' | 'raise' | 'call' | 'print' | 'log',
  over: 'warn' as 'ignore' | 'warn' | 'raise' | 'call' | 'print' | 'log',
  under: 'ignore' as 'ignore' | 'warn' | 'raise' | 'call' | 'print' | 'log',
  invalid: 'warn' as 'ignore' | 'warn' | 'raise' | 'call' | 'print' | 'log',
};

type ErrorMode = 'ignore' | 'warn' | 'raise' | 'call' | 'print' | 'log';

export interface FloatErrorState {
  divide: ErrorMode;
  over: ErrorMode;
  under: ErrorMode;
  invalid: ErrorMode;
}

/**
 * Get the current floating-point error handling.
 *
 * @returns Current error handling settings
 */
export function geterr(): FloatErrorState {
  return { ..._floatErrorState };
}

/**
 * Set how floating-point errors are handled.
 *
 * @param all - Set all error modes at once (optional)
 * @param divide - Treatment for division by zero
 * @param over - Treatment for floating-point overflow
 * @param under - Treatment for floating-point underflow
 * @param invalid - Treatment for invalid floating-point operation
 * @returns Previous error handling settings
 */
export function seterr(
  all?: ErrorMode,
  divide?: ErrorMode,
  over?: ErrorMode,
  under?: ErrorMode,
  invalid?: ErrorMode
): FloatErrorState {
  const old = geterr();

  if (all !== undefined) {
    _floatErrorState.divide = all;
    _floatErrorState.over = all;
    _floatErrorState.under = all;
    _floatErrorState.invalid = all;
  }

  if (divide !== undefined) _floatErrorState.divide = divide;
  if (over !== undefined) _floatErrorState.over = over;
  if (under !== undefined) _floatErrorState.under = under;
  if (invalid !== undefined) _floatErrorState.invalid = invalid;

  return old;
}
