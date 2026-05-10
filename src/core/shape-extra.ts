/**
 * Additional shape manipulation functions
 *
 * Tree-shakeable standalone functions for array manipulation.
 */

import { Complex } from '../common/complex';
import { hasFloat16, isComplexDType } from '../common/dtype';
import { type DType, NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import { wasmPad2D } from '../common/wasm/pad';
import { array, zeros } from './creation';
import { concatenate, flatten } from './shape';

/**
 * Append values to the end of an array
 */
export function append(
  arr: NDArrayCore,
  values: NDArrayCore | number | number[],
  axis?: number,
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
  const srcStorage = arr.storage;

  if (axis === undefined) {
    // Flatten and delete
    const flat = flatten(arr);
    const flatStorage = flat.storage;
    const newData: number[] = [];

    const deleteSet = new Set(indices.map((i) => (i < 0 ? flat.size + i : i)));

    for (let i = 0; i < flat.size; i++) {
      if (!deleteSet.has(i)) {
        newData.push(flatStorage.iget(i) as number);
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
  const resultStorage = result.storage;

  // Calculate strides for multi-index iteration (logical C-contiguous strides)
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

    resultStorage.iset(newFlatIdx, srcStorage.iget(flatIdx));
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
  axis?: number,
): NDArrayCore {
  const indices = Array.isArray(obj) ? obj : [obj];
  const valuesArr =
    values instanceof NDArrayCore ? values : array(Array.isArray(values) ? values : [values]);

  if (axis === undefined) {
    // Flatten and insert
    const flat = flatten(arr);
    const flatValues = flatten(valuesArr);
    const dtype = arr.dtype as DType;
    const isComplex = isComplexDType(dtype);

    if (isComplex) {
      // Complex path: work with Complex objects via iget/iset
      const flatSize = flat.size;
      const valSize = flatValues.size;
      const items: Complex[] = [];
      for (let i = 0; i < flatSize; i++) {
        items.push(flat.storage.iget(i) as Complex);
      }

      if (indices.length === 1) {
        const rawIdx = indices[0]!;
        const idx = rawIdx < 0 ? flatSize + rawIdx : rawIdx;
        const vals: Complex[] = [];
        for (let i = 0; i < valSize; i++) {
          const v = flatValues.storage.iget(i);
          vals.push(v instanceof Complex ? v : new Complex(Number(v), 0));
        }
        items.splice(idx, 0, ...vals);
      } else {
        const indexPairs = indices
          .map((rawIdx, i) => ({
            idx: rawIdx < 0 ? flatSize + rawIdx : rawIdx,
            valIdx: i,
          }))
          .sort((a, b) => a.idx - b.idx);

        for (let i = 0; i < indexPairs.length; i++) {
          const { idx, valIdx } = indexPairs[i]!;
          const v = flatValues.storage.iget(valIdx % valSize);
          const val = v instanceof Complex ? v : new Complex(Number(v), 0);
          items.splice(idx + i, 0, val);
        }
      }

      const resultStorage = ArrayStorage.empty([items.length], dtype);
      const resultData = resultStorage.data as Float64Array | Float32Array;
      for (let i = 0; i < items.length; i++) {
        resultData[i * 2] = items[i]!.re;
        resultData[i * 2 + 1] = items[i]!.im;
      }
      return new NDArrayCore(resultStorage);
    }

    const flatData = flat.data;
    const valuesData = flatValues.data;
    const result: number[] = Array.from(flatData as unknown as ArrayLike<number>);

    if (indices.length === 1) {
      // Single index: insert all values at that position
      const rawIdx = indices[0]!;
      const idx = rawIdx < 0 ? flat.size + rawIdx : rawIdx;
      const vals: number[] = Array.from(valuesData as unknown as ArrayLike<number>);
      result.splice(idx, 0, ...vals);
    } else {
      // Multiple indices: insert one value per index (cycling through values)
      // Sort indices ascending for correct offset tracking
      const indexPairs = indices
        .map((rawIdx, i) => ({
          idx: rawIdx < 0 ? flat.size + rawIdx : rawIdx,
          valIdx: i,
        }))
        .sort((a, b) => a.idx - b.idx);

      for (let i = 0; i < indexPairs.length; i++) {
        const { idx, valIdx } = indexPairs[i]!;
        const val = valuesData[valIdx % valuesData.length] as number;
        result.splice(idx + i, 0, val);
      }
    }

    return array(result, dtype);
  }

  // Insert along axis - simplified implementation
  throw new Error('insert along axis not fully implemented in standalone');
}

export type PadWidthArg = number | [number, number] | (number | [number, number])[];
export type PadValueArg = number | [number, number] | (number | [number, number])[];

/**
 * Normalize a pad_width / constant_values argument to per-axis [before, after] pairs.
 *
 * Accepts (matching NumPy):
 *  - scalar `n` → `[[n, n], ...]` for every axis
 *  - `[n]` (length 1) → `[[n, n], ...]` for every axis
 *  - `[before, after]` (length 2) → broadcast to every axis
 *  - `[n0, n1, ..., n_{ndim-1}]` (length ndim) → per-axis scalars (before=after=n_k)
 *  - `[[b, a]]` (length 1) → broadcast pair to every axis
 *  - `[[b0, a0], ..., [b_{ndim-1}, a_{ndim-1}]]` (length ndim) → per-axis pairs
 *  - mixed: `[n0, [b1, a1], ...]` — scalars expand to (n, n)
 */
function normalizePerAxisPair(
  v: PadWidthArg,
  ndim: number,
  paramName: string,
): [number, number][] {
  if (typeof v === 'number') {
    return Array.from({ length: ndim }, () => [v, v] as [number, number]);
  }
  if (!Array.isArray(v) || v.length === 0) {
    throw new Error(`${paramName} must be a number, pair, or non-empty list`);
  }

  // Distinguish "list of scalars" from "list of pairs (or mixed)"
  const allNumbers = v.every((x) => typeof x === 'number');
  if (allNumbers) {
    const nums = v as number[];
    if (nums.length === 1) {
      const n = nums[0]!;
      return Array.from({ length: ndim }, () => [n, n] as [number, number]);
    }
    if (nums.length === 2) {
      const pair: [number, number] = [nums[0]!, nums[1]!];
      return Array.from({ length: ndim }, () => [...pair] as [number, number]);
    }
    if (nums.length === ndim) {
      return nums.map((n) => [n, n] as [number, number]);
    }
    throw new Error(`${paramName} length ${nums.length} does not match ndim ${ndim}`);
  }

  // List of pairs (or mixed scalars/pairs)
  const toPair = (item: number | [number, number]): [number, number] =>
    typeof item === 'number' ? [item, item] : ([item[0]!, item[1]!] as [number, number]);

  if (v.length === 1) {
    const pair = toPair(v[0]!);
    return Array.from({ length: ndim }, () => [...pair] as [number, number]);
  }
  if (v.length !== ndim) {
    throw new Error(`${paramName} length ${v.length} does not match ndim ${ndim}`);
  }
  return v.map(toPair);
}

/**
 * Pad an array
 */
export function pad(
  arr: NDArrayCore,
  pad_width: PadWidthArg,
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
  constant_values: PadValueArg = 0,
): NDArrayCore {
  const shape = [...arr.shape];
  const ndim = shape.length;

  const padWidths = normalizePerAxisPair(pad_width, ndim, 'pad_width');
  const padValues = normalizePerAxisPair(constant_values, ndim, 'constant_values');

  // Calculate new shape
  const newShape = shape.map((s, i) => s + padWidths[i]![0] + padWidths[i]![1]);

  // True if every axis uses (constantValue, constantValue) — enables the
  // simple "pre-fill then copy source" fast paths below.
  const uniformValue = padValues.every(
    (p) => p[0] === padValues[0]![0] && p[1] === padValues[0]![0],
  );
  const scalarFillValue = uniformValue ? padValues[0]![0] : 0;

  // WASM fast path for 2D uniform zero-pad
  if (
    mode === 'constant' &&
    uniformValue &&
    scalarFillValue === 0 &&
    ndim === 2 &&
    typeof pad_width === 'number'
  ) {
    const wasm = wasmPad2D(arr.storage, pad_width);
    if (wasm) {
      return new NDArrayCore(wasm);
    }
  }

  if (mode !== 'constant') {
    throw new Error(`pad mode '${mode}' not fully implemented in standalone`);
  }

  const result = zeros(newShape, arr.dtype as DType);
  const resultStorage = result.storage;
  const srcStorage = arr.storage;

  // Strides for multi-index iteration (logical C-contiguous)
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
  const totalSize = shape.reduce((a, b) => a * b, 1);

  // Per-axis value path: paint output cell-by-cell. NumPy applies pad axis-by-axis
  // in order, so the *highest* axis whose pad covers a given cell wins.
  if (!uniformValue) {
    const newSize = result.size;
    for (let outFlat = 0; outFlat < newSize; outFlat++) {
      const idx: number[] = new Array(ndim);
      let remaining = outFlat;
      for (let k = 0; k < ndim; k++) {
        idx[k] = Math.floor(remaining / newStrides[k]!);
        remaining %= newStrides[k]!;
      }

      // Find highest axis whose pad covers this cell.
      let fillValue: number | undefined;
      for (let k = ndim - 1; k >= 0; k--) {
        const i = idx[k]!;
        const pb = padWidths[k]![0];
        const sz = shape[k]!;
        if (i < pb) {
          fillValue = padValues[k]![0];
          break;
        }
        if (i >= pb + sz) {
          fillValue = padValues[k]![1];
          break;
        }
      }

      if (fillValue !== undefined) {
        resultStorage.iset(outFlat, fillValue);
      } else {
        // In source region; copy from input.
        let srcFlat = 0;
        for (let k = 0; k < ndim; k++) {
          srcFlat += (idx[k]! - padWidths[k]![0]) * strides[k]!;
        }
        resultStorage.iset(outFlat, srcStorage.iget(srcFlat));
      }
    }
    return result;
  }

  // Uniform-value fast paths below (mirror previous behavior).
  if (scalarFillValue !== 0) {
    for (let i = 0; i < result.size; i++) {
      resultStorage.iset(i, scalarFillValue);
    }
  }

  // Float16Array optimization: bulk-convert for faster per-element access
  if (
    arr.dtype === 'float16' &&
    hasFloat16 &&
    srcStorage.isCContiguous &&
    resultStorage.isCContiguous
  ) {
    const f32Src = new Float32Array(
      (srcStorage.data as Float16Array).subarray(srcStorage.offset, srcStorage.offset + totalSize),
    );
    const resultSize = resultStorage.size;
    const f32Result = new Float32Array(resultSize);
    if (scalarFillValue !== 0) {
      f32Result.fill(scalarFillValue);
    }

    for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
      const multiIdx: number[] = [];
      let remaining = flatIdx;
      for (let i = 0; i < shape.length; i++) {
        multiIdx.push(Math.floor(remaining / strides[i]!));
        remaining = remaining % strides[i]!;
      }

      let newFlatIdx = 0;
      for (let i = 0; i < newShape.length; i++) {
        newFlatIdx += (multiIdx[i]! + padWidths[i]![0]) * newStrides[i]!;
      }

      f32Result[newFlatIdx] = f32Src[flatIdx]!;
    }

    (resultStorage.data as Float16Array).set(f32Result);
    return result;
  }

  for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
    const multiIdx: number[] = [];
    let remaining = flatIdx;
    for (let i = 0; i < shape.length; i++) {
      multiIdx.push(Math.floor(remaining / strides[i]!));
      remaining = remaining % strides[i]!;
    }

    let newFlatIdx = 0;
    for (let i = 0; i < newShape.length; i++) {
      newFlatIdx += (multiIdx[i]! + padWidths[i]![0]) * newStrides[i]!;
    }

    resultStorage.iset(newFlatIdx, srcStorage.iget(flatIdx));
  }

  return result;
}
