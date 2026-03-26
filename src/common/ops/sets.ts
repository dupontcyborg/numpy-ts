/**
 * Set operations
 */

import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';

// Helper: compare complex numbers lexicographically
function complexCompare(aRe: number, aIm: number, bRe: number, bIm: number): number {
  const aIsNaN = isNaN(aRe) || isNaN(aIm);
  const bIsNaN = isNaN(bRe) || isNaN(bIm);
  if (aIsNaN && bIsNaN) return 0;
  if (aIsNaN) return 1;
  if (bIsNaN) return -1;
  if (aRe < bRe) return -1;
  if (aRe > bRe) return 1;
  if (aIm < bIm) return -1;
  if (aIm > bIm) return 1;
  return 0;
}

// Helper: check if two complex numbers are equal
function complexEqual(aRe: number, aIm: number, bRe: number, bIm: number): boolean {
  const aIsNaN = isNaN(aRe) || isNaN(aIm);
  const bIsNaN = isNaN(bRe) || isNaN(bIm);
  if (aIsNaN && bIsNaN) return true; // Both NaN are considered equal for uniqueness
  if (aIsNaN || bIsNaN) return false;
  return aRe === bRe && aIm === bIm;
}

/**
 * Counting sort path for small-range integer types.
 * O(n + range) instead of O(n log n).
 */
function uniqueCountingSort(
  data: TypedArray,
  off: number,
  size: number,
  dtype: DType,
  minVal: number,
  range: number,
  returnIndex: boolean,
  returnInverse: boolean,
  returnCounts: boolean
):
  | ArrayStorage
  | {
      values: ArrayStorage;
      indices?: ArrayStorage;
      inverse?: ArrayStorage;
      counts?: ArrayStorage;
    } {
  const bucketCounts = new Int32Array(range);
  const firstIdx = returnIndex ? new Int32Array(range).fill(-1) : null;

  for (let i = 0; i < size; i++) {
    const v = Number((data as Int8Array)[off + i]!) - minVal;
    bucketCounts[v]!++;
    if (firstIdx !== null && firstIdx[v] === -1) firstIdx[v] = i;
  }

  let numUnique = 0;
  for (let v = 0; v < range; v++) {
    if (bucketCounts[v]! > 0) numUnique++;
  }

  // Allocate result arrays directly — no intermediate copies
  const uniqueResult = ArrayStorage.zeros([numUnique], dtype);
  const uniqueData = uniqueResult.data;
  const isBigInt = uniqueData instanceof BigInt64Array || uniqueData instanceof BigUint64Array;

  const countsResult = returnCounts ? ArrayStorage.zeros([numUnique], 'int32') : null;
  const countsData = countsResult ? (countsResult.data as Int32Array) : null;
  const indicesResult = returnIndex ? ArrayStorage.zeros([numUnique], 'int32') : null;
  const indicesData = indicesResult ? (indicesResult.data as Int32Array) : null;
  const valToUniqueIdx = returnInverse ? new Int32Array(range).fill(-1) : null;

  let ui = 0;
  for (let v = 0; v < range; v++) {
    if (bucketCounts[v]! > 0) {
      if (isBigInt) {
        (uniqueData as BigInt64Array)[ui] = BigInt(v + minVal);
      } else {
        (uniqueData as Int8Array)[ui] = v + minVal;
      }
      if (countsData) countsData[ui] = bucketCounts[v]!;
      if (indicesData) indicesData[ui] = firstIdx![v]!;
      if (valToUniqueIdx) valToUniqueIdx[v] = ui;
      ui++;
    }
  }

  if (!returnIndex && !returnInverse && !returnCounts) return uniqueResult;

  const result: {
    values: ArrayStorage;
    indices?: ArrayStorage;
    inverse?: ArrayStorage;
    counts?: ArrayStorage;
  } = { values: uniqueResult };

  if (indicesResult) result.indices = indicesResult;

  if (returnInverse) {
    const inverseResult = ArrayStorage.zeros([size], 'int32');
    const inverseData = inverseResult.data as Int32Array;
    for (let i = 0; i < size; i++) {
      inverseData[i] = valToUniqueIdx![Number((data as Int8Array)[off + i]!) - minVal]!;
    }
    result.inverse = inverseResult;
  }

  if (countsResult) result.counts = countsResult;

  return result;
}

/**
 * Find the unique elements of an array
 */
export function unique(
  a: ArrayStorage,
  returnIndex: boolean = false,
  returnInverse: boolean = false,
  returnCounts: boolean = false,
  axis?: number
):
  | ArrayStorage
  | {
      values: ArrayStorage;
      indices?: ArrayStorage;
      inverse?: ArrayStorage;
      counts?: ArrayStorage;
    } {
  // Axis-based unique: find unique slices along the given axis
  if (axis !== undefined) {
    const shape = Array.from(a.shape);
    const ndim = shape.length;
    let normalizedAxis = axis < 0 ? ndim + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`unique: axis ${axis} out of bounds for array of dimension ${ndim}`);
    }
    const axisSize = shape[normalizedAxis]!;
    // Serialize each slice along the axis to a comparable key
    const sliceKeys: string[] = [];
    for (let i = 0; i < axisSize; i++) {
      // Build a key by collecting all elements in the slice
      const parts: number[] = [];
      // Iterate over all indices with this i on the given axis
      const otherShape = shape.filter((_, d) => d !== normalizedAxis);
      const otherSize = otherShape.reduce((acc, s) => acc * s, 1);
      for (let j = 0; j < otherSize; j++) {
        // Convert j to multi-index in otherShape
        let rem = j;
        const otherIdx: number[] = new Array(otherShape.length);
        for (let d = otherShape.length - 1; d >= 0; d--) {
          otherIdx[d] = rem % otherShape[d]!;
          rem = Math.floor(rem / otherShape[d]!);
        }
        // Build full index inserting i at normalizedAxis
        const fullIdx: number[] = [];
        let oi = 0;
        for (let d = 0; d < ndim; d++) {
          fullIdx.push(d === normalizedAxis ? i : otherIdx[oi++]!);
        }
        parts.push(Number(a.get(...fullIdx)));
      }
      sliceKeys.push(parts.join(','));
    }
    // Find unique slice indices (sorted lexicographically)
    const indexedKeys = sliceKeys.map((key, i) => ({ key, i }));
    indexedKeys.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    const uniqueIndices: number[] = [];
    let lastKey: string | undefined = undefined;
    for (const { key, i } of indexedKeys) {
      if (key !== lastKey) {
        uniqueIndices.push(i);
        lastKey = key;
      }
    }
    // Build output array with unique slices stacked
    const outShape = shape.map((s, d) => (d === normalizedAxis ? uniqueIndices.length : s));
    const result = ArrayStorage.zeros(outShape, a.dtype);
    for (let ui = 0; ui < uniqueIndices.length; ui++) {
      const srcIdx = uniqueIndices[ui]!;
      const otherShape = shape.filter((_, d) => d !== normalizedAxis);
      const otherSize = otherShape.reduce((acc, s) => acc * s, 1);
      for (let j = 0; j < otherSize; j++) {
        let rem = j;
        const otherIdx: number[] = new Array(otherShape.length);
        for (let d = otherShape.length - 1; d >= 0; d--) {
          otherIdx[d] = rem % otherShape[d]!;
          rem = Math.floor(rem / otherShape[d]!);
        }
        const srcFullIdx: number[] = [];
        const dstFullIdx: number[] = [];
        let oi = 0;
        for (let d = 0; d < ndim; d++) {
          srcFullIdx.push(d === normalizedAxis ? srcIdx : otherIdx[oi]!);
          dstFullIdx.push(d === normalizedAxis ? ui : otherIdx[oi]!);
          if (d !== normalizedAxis) oi++;
        }
        result.set(dstFullIdx, Number(a.get(...srcFullIdx)));
      }
    }
    return result;
  }

  const dtype = a.dtype;
  const size = a.size;
  const data = a.data;
  const off = a.offset;

  // Complex unique with lexicographic ordering
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    // Collect complex values with original indices
    const values: { re: number; im: number; index: number }[] = [];
    for (let i = 0; i < size; i++) {
      values.push({
        re: complexData[(off + i) * 2]!,
        im: complexData[(off + i) * 2 + 1]!,
        index: i,
      });
    }

    // Sort by lexicographic order
    values.sort((x, y) => complexCompare(x.re, x.im, y.re, y.im));

    // Find unique values
    const uniqueValues: { re: number; im: number }[] = [];
    const indices: number[] = [];
    const inverse: number[] = new Array(size);
    const counts: number[] = [];

    let lastRe: number | undefined = undefined;
    let lastIm: number | undefined = undefined;
    let currentCount = 0;

    for (let i = 0; i < values.length; i++) {
      const { re, im, index } = values[i]!;
      const isDifferent = lastRe === undefined || !complexEqual(re, im, lastRe!, lastIm!);

      if (isDifferent) {
        if (lastRe !== undefined) {
          counts.push(currentCount);
        }
        uniqueValues.push({ re, im });
        indices.push(index);
        currentCount = 1;
        lastRe = re;
        lastIm = im;
      } else {
        currentCount++;
      }
    }
    if (currentCount > 0) {
      counts.push(currentCount);
    }

    // Build inverse mapping using a key string
    const valueToUniqueIdx = new Map<string, number>();
    let nanIdx = -1;
    for (let i = 0; i < uniqueValues.length; i++) {
      const { re, im } = uniqueValues[i]!;
      if (isNaN(re) || isNaN(im)) {
        nanIdx = i;
      } else {
        valueToUniqueIdx.set(`${re},${im}`, i);
      }
    }
    for (let i = 0; i < size; i++) {
      const re = complexData[(off + i) * 2]!;
      const im = complexData[(off + i) * 2 + 1]!;
      if (isNaN(re) || isNaN(im)) {
        inverse[i] = nanIdx;
      } else {
        inverse[i] = valueToUniqueIdx.get(`${re},${im}`)!;
      }
    }

    // Create result arrays
    const uniqueResult = ArrayStorage.zeros([uniqueValues.length], dtype);
    const uniqueData = uniqueResult.data as Float64Array | Float32Array;
    for (let i = 0; i < uniqueValues.length; i++) {
      uniqueData[i * 2] = uniqueValues[i]!.re;
      uniqueData[i * 2 + 1] = uniqueValues[i]!.im;
    }

    if (!returnIndex && !returnInverse && !returnCounts) {
      return uniqueResult;
    }

    const result: {
      values: ArrayStorage;
      indices?: ArrayStorage;
      inverse?: ArrayStorage;
      counts?: ArrayStorage;
    } = { values: uniqueResult };

    if (returnIndex) {
      const indicesResult = ArrayStorage.zeros([indices.length], 'int32');
      const indicesData = indicesResult.data as Int32Array;
      for (let i = 0; i < indices.length; i++) {
        indicesData[i] = indices[i]!;
      }
      result.indices = indicesResult;
    }

    if (returnInverse) {
      const inverseResult = ArrayStorage.zeros([inverse.length], 'int32');
      const inverseData = inverseResult.data as Int32Array;
      for (let i = 0; i < inverse.length; i++) {
        inverseData[i] = inverse[i]!;
      }
      result.inverse = inverseResult;
    }

    if (returnCounts) {
      const countsResult = ArrayStorage.zeros([counts.length], 'int32');
      const countsData = countsResult.data as Int32Array;
      for (let i = 0; i < counts.length; i++) {
        countsData[i] = counts[i]!;
      }
      result.counts = countsResult;
    }

    return result;
  }

  // --- Non-complex, non-axis unique ---

  // Fast path: counting sort for int8/uint8 (range 256, always worth it)
  if (dtype === 'int8' || dtype === 'uint8') {
    const minVal = dtype === 'int8' ? -128 : 0;
    return uniqueCountingSort(
      data,
      off,
      size,
      dtype,
      minVal,
      256,
      returnIndex,
      returnInverse,
      returnCounts
    );
  }

  // For int16/uint16/int32/uint32: counting sort only if range is small relative to size
  if (dtype === 'int16' || dtype === 'uint16' || dtype === 'int32' || dtype === 'uint32') {
    let min = Number(data[off]!),
      max = min;
    for (let i = 1; i < size; i++) {
      const v = Number(data[off + i]!);
      if (v < min) min = v;
      else if (v > max) max = v;
    }
    const range = max - min + 1;
    if (range <= 4 * size) {
      return uniqueCountingSort(
        data,
        off,
        size,
        dtype,
        min,
        range,
        returnIndex,
        returnInverse,
        returnCounts
      );
    }
  }

  const isBigInt = data instanceof BigInt64Array || data instanceof BigUint64Array;
  const isFloat = dtype === 'float64' || dtype === 'float32' || dtype === 'float16';

  // Fast path: when we don't need original indices, sort values directly
  if (!returnIndex && !returnInverse) {
    const vals = new Float64Array(size);
    for (let i = 0; i < size; i++) vals[i] = Number(data[off + i]!);
    vals.sort(); // native optimized sort, NaN goes to end

    if (size === 0) {
      const empty = ArrayStorage.zeros([0], dtype as DType);
      return returnCounts ? { values: empty, counts: ArrayStorage.zeros([0], 'int32') } : empty;
    }

    // Walk sorted array to find unique values and counts
    const uniqueVals: number[] = [vals[0]!];
    const counts: number[] = [1];

    for (let i = 1; i < size; i++) {
      const v = vals[i]!;
      const prev = vals[i - 1]!;
      // After sort, NaN clusters at end; treat consecutive NaN as same
      if (v !== prev && !(v !== v && prev !== prev)) {
        uniqueVals.push(v);
        counts.push(1);
      } else {
        counts[counts.length - 1]!++;
      }
    }

    const numUnique = uniqueVals.length;
    const uniqueResult = ArrayStorage.zeros([numUnique], dtype as DType);
    const uniqueData = uniqueResult.data;
    if (isBigInt) {
      for (let i = 0; i < numUnique; i++) {
        (uniqueData as BigInt64Array)[i] = BigInt(uniqueVals[i]!);
      }
    } else {
      for (let i = 0; i < numUnique; i++) {
        (uniqueData as Float64Array)[i] = uniqueVals[i]!;
      }
    }

    if (!returnCounts) return uniqueResult;

    const countsResult = ArrayStorage.zeros([numUnique], 'int32');
    const countsData = countsResult.data as Int32Array;
    for (let i = 0; i < numUnique; i++) countsData[i] = counts[i]!;
    return { values: uniqueResult, counts: countsResult };
  }

  // General path: need to track original indices (returnIndex or returnInverse)
  // Use parallel typed arrays instead of object array to avoid GC pressure
  const vals = new Float64Array(size);
  const sortedIdxs = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    vals[i] = Number(data[off + i]!);
    sortedIdxs[i] = i;
  }

  // Sort index array by value
  if (isFloat) {
    sortedIdxs.sort((a, b) => {
      const av = vals[a]!,
        bv = vals[b]!;
      // NaN check using self-inequality (faster than isNaN)
      if (av !== av && bv !== bv) return 0;
      if (av !== av) return 1;
      if (bv !== bv) return -1;
      return av - bv;
    });
  } else {
    // Integer types: no NaN possible
    sortedIdxs.sort((a, b) => vals[a]! - vals[b]!);
  }

  // Find unique values
  const uniqueValues: number[] = [];
  const findIndices: number[] = [];
  const countsArr: number[] = [];

  let lastValue: number | undefined;
  let currentCount = 0;

  for (let i = 0; i < size; i++) {
    const idx = sortedIdxs[i]!;
    const value = vals[idx]!;
    const isDifferent =
      lastValue === undefined ||
      (isFloat
        ? (value !== value && lastValue === lastValue) ||
          (value === value && lastValue !== lastValue) ||
          (value === value && value !== lastValue)
        : value !== lastValue);

    if (isDifferent) {
      if (lastValue !== undefined) countsArr.push(currentCount);
      uniqueValues.push(value);
      findIndices.push(idx);
      currentCount = 1;
      lastValue = value;
    } else {
      currentCount++;
    }
  }
  if (currentCount > 0) countsArr.push(currentCount);

  // Create result arrays
  const numUnique = uniqueValues.length;
  const uniqueResult = ArrayStorage.zeros([numUnique], dtype as DType);
  const uniqueData = uniqueResult.data;
  const resultIsBigInt =
    uniqueData instanceof BigInt64Array || uniqueData instanceof BigUint64Array;
  for (let i = 0; i < numUnique; i++) {
    (uniqueData as Float64Array | Int32Array | BigInt64Array)[i] = resultIsBigInt
      ? BigInt(uniqueValues[i]!)
      : (uniqueValues[i]! as never);
  }

  if (!returnIndex && !returnInverse && !returnCounts) return uniqueResult;

  const result: {
    values: ArrayStorage;
    indices?: ArrayStorage;
    inverse?: ArrayStorage;
    counts?: ArrayStorage;
  } = { values: uniqueResult };

  if (returnIndex) {
    const indicesResult = ArrayStorage.zeros([numUnique], 'int32');
    const indicesData = indicesResult.data as Int32Array;
    for (let i = 0; i < numUnique; i++) indicesData[i] = findIndices[i]!;
    result.indices = indicesResult;
  }

  if (returnInverse) {
    // Build inverse mapping only when needed
    const valueToUniqueIdx = new Map<number, number>();
    let nanIdx = -1;
    for (let i = 0; i < numUnique; i++) {
      const v = uniqueValues[i]!;
      if (v !== v) nanIdx = i;
      else valueToUniqueIdx.set(v, i);
    }
    const inverseResult = ArrayStorage.zeros([size], 'int32');
    const inverseData = inverseResult.data as Int32Array;
    for (let i = 0; i < size; i++) {
      const val = vals[i]!;
      inverseData[i] = val !== val ? nanIdx : valueToUniqueIdx.get(val)!;
    }
    result.inverse = inverseResult;
  }

  if (returnCounts) {
    const countsResult = ArrayStorage.zeros([numUnique], 'int32');
    const countsData = countsResult.data as Int32Array;
    for (let i = 0; i < numUnique; i++) countsData[i] = countsArr[i]!;
    result.counts = countsResult;
  }

  return result;
}

// Helper: convert storage element to key string
function elementToKey(
  data: ArrayLike<number | bigint>,
  index: number,
  isComplex: boolean,
  offset: number = 0
): string {
  if (isComplex) {
    const re = Number((data as Float64Array)[(offset + index) * 2]);
    const im = Number((data as Float64Array)[(offset + index) * 2 + 1]);
    return `${re},${im}`;
  }
  return String(Number(data[offset + index]!));
}

/**
 * Test whether each element of a 1-D array is also present in a second array
 */
export function in1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  return isin(ar1, ar2);
}

/**
 * Find the intersection of two arrays
 */
export function intersect1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;
  const isComplex = isComplexDType(dtype);

  const unique1 = unique(ar1) as ArrayStorage;
  const unique2 = unique(ar2) as ArrayStorage;

  const set2 = new Set<string>();
  for (let i = 0; i < unique2.size; i++) {
    set2.add(elementToKey(unique2.data, i, isComplex));
  }

  const intersectionIndices: number[] = [];
  for (let i = 0; i < unique1.size; i++) {
    const key = elementToKey(unique1.data, i, isComplex);
    if (set2.has(key)) {
      intersectionIndices.push(i);
    }
  }

  // unique1 is already sorted, just copy the matching elements
  if (isComplex) {
    const result = ArrayStorage.zeros([intersectionIndices.length], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const unique1Data = unique1.data as Float64Array | Float32Array;
    for (let i = 0; i < intersectionIndices.length; i++) {
      const idx = intersectionIndices[i]!;
      resultData[i * 2] = unique1Data[idx * 2]!;
      resultData[i * 2 + 1] = unique1Data[idx * 2 + 1]!;
    }
    return result;
  }

  const result = ArrayStorage.zeros([intersectionIndices.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < intersectionIndices.length; i++) {
    resultData[i] = unique1.data[intersectionIndices[i]!]!;
  }
  return result;
}

/**
 * Test whether each element of an ND array is also present in a second array
 */
export function isin(element: ArrayStorage, testElements: ArrayStorage): ArrayStorage {
  const shape = Array.from(element.shape);
  const size = element.size;
  const isComplex = isComplexDType(element.dtype);

  const testSet = new Set<string>();
  for (let i = 0; i < testElements.size; i++) {
    testSet.add(elementToKey(testElements.data, i, isComplex, testElements.offset));
  }

  const result = ArrayStorage.zeros(shape, 'bool');
  const resultData = result.data as Uint8Array;

  for (let i = 0; i < size; i++) {
    const key = elementToKey(element.data, i, isComplex, element.offset);
    resultData[i] = testSet.has(key) ? 1 : 0;
  }

  return result;
}

/**
 * Find the set difference of two arrays
 */
export function setdiff1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;
  const isComplex = isComplexDType(dtype);

  const unique1 = unique(ar1) as ArrayStorage;

  const set2 = new Set<string>();
  for (let i = 0; i < ar2.size; i++) {
    set2.add(elementToKey(ar2.data, i, isComplex, ar2.offset));
  }

  const diffIndices: number[] = [];
  for (let i = 0; i < unique1.size; i++) {
    const key = elementToKey(unique1.data, i, isComplex);
    if (!set2.has(key)) {
      diffIndices.push(i);
    }
  }

  if (isComplex) {
    const result = ArrayStorage.zeros([diffIndices.length], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const unique1Data = unique1.data as Float64Array | Float32Array;
    for (let i = 0; i < diffIndices.length; i++) {
      const idx = diffIndices[i]!;
      resultData[i * 2] = unique1Data[idx * 2]!;
      resultData[i * 2 + 1] = unique1Data[idx * 2 + 1]!;
    }
    return result;
  }

  const result = ArrayStorage.zeros([diffIndices.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < diffIndices.length; i++) {
    resultData[i] = unique1.data[diffIndices[i]!]!;
  }
  return result;
}

/**
 * Find the set exclusive-or of two arrays
 */
export function setxor1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;
  const isComplex = isComplexDType(dtype);

  const unique1 = unique(ar1) as ArrayStorage;
  const unique2 = unique(ar2) as ArrayStorage;

  const set1 = new Set<string>();
  const set2 = new Set<string>();

  for (let i = 0; i < unique1.size; i++) {
    set1.add(elementToKey(unique1.data, i, isComplex));
  }
  for (let i = 0; i < unique2.size; i++) {
    set2.add(elementToKey(unique2.data, i, isComplex));
  }

  // Collect from unique1 not in set2, and unique2 not in set1
  const xorIndices1: number[] = [];
  const xorIndices2: number[] = [];

  for (let i = 0; i < unique1.size; i++) {
    const key = elementToKey(unique1.data, i, isComplex);
    if (!set2.has(key)) {
      xorIndices1.push(i);
    }
  }
  for (let i = 0; i < unique2.size; i++) {
    const key = elementToKey(unique2.data, i, isComplex);
    if (!set1.has(key)) {
      xorIndices2.push(i);
    }
  }

  if (isComplex) {
    // Collect all values, then sort
    const xorValues: { re: number; im: number }[] = [];
    const u1Data = unique1.data as Float64Array | Float32Array;
    const u2Data = unique2.data as Float64Array | Float32Array;

    for (const idx of xorIndices1) {
      xorValues.push({ re: u1Data[idx * 2]!, im: u1Data[idx * 2 + 1]! });
    }
    for (const idx of xorIndices2) {
      xorValues.push({ re: u2Data[idx * 2]!, im: u2Data[idx * 2 + 1]! });
    }

    xorValues.sort((a, b) => complexCompare(a.re, a.im, b.re, b.im));

    const result = ArrayStorage.zeros([xorValues.length], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    for (let i = 0; i < xorValues.length; i++) {
      resultData[i * 2] = xorValues[i]!.re;
      resultData[i * 2 + 1] = xorValues[i]!.im;
    }
    return result;
  }

  // Collect all values, then sort
  const xorValues: number[] = [];
  for (const idx of xorIndices1) {
    xorValues.push(Number(unique1.data[idx]!));
  }
  for (const idx of xorIndices2) {
    xorValues.push(Number(unique2.data[idx]!));
  }

  xorValues.sort((a, b) => {
    if (isNaN(a) && isNaN(b)) return 0;
    if (isNaN(a)) return 1;
    if (isNaN(b)) return -1;
    return a - b;
  });

  const result = ArrayStorage.zeros([xorValues.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < xorValues.length; i++) {
    resultData[i] = xorValues[i]!;
  }
  return result;
}

/**
 * Find the union of two arrays
 */
export function union1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;
  const isComplex = isComplexDType(dtype);

  const unique1 = unique(ar1) as ArrayStorage;
  const unique2 = unique(ar2) as ArrayStorage;

  const unionSet = new Set<string>();
  const unionValues: { re: number; im: number }[] = [];

  if (isComplex) {
    const u1Data = unique1.data as Float64Array | Float32Array;
    const u2Data = unique2.data as Float64Array | Float32Array;

    for (let i = 0; i < unique1.size; i++) {
      const re = u1Data[i * 2]!;
      const im = u1Data[i * 2 + 1]!;
      const key = `${re},${im}`;
      if (!unionSet.has(key)) {
        unionSet.add(key);
        unionValues.push({ re, im });
      }
    }
    for (let i = 0; i < unique2.size; i++) {
      const re = u2Data[i * 2]!;
      const im = u2Data[i * 2 + 1]!;
      const key = `${re},${im}`;
      if (!unionSet.has(key)) {
        unionSet.add(key);
        unionValues.push({ re, im });
      }
    }

    unionValues.sort((a, b) => complexCompare(a.re, a.im, b.re, b.im));

    const result = ArrayStorage.zeros([unionValues.length], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    for (let i = 0; i < unionValues.length; i++) {
      resultData[i * 2] = unionValues[i]!.re;
      resultData[i * 2 + 1] = unionValues[i]!.im;
    }
    return result;
  }

  const realValues: number[] = [];
  for (let i = 0; i < unique1.size; i++) {
    const val = Number(unique1.data[i]!);
    const key = String(val);
    if (!unionSet.has(key)) {
      unionSet.add(key);
      realValues.push(val);
    }
  }
  for (let i = 0; i < unique2.size; i++) {
    const val = Number(unique2.data[i]!);
    const key = String(val);
    if (!unionSet.has(key)) {
      unionSet.add(key);
      realValues.push(val);
    }
  }

  realValues.sort((a, b) => {
    if (isNaN(a) && isNaN(b)) return 0;
    if (isNaN(a)) return 1;
    if (isNaN(b)) return -1;
    return a - b;
  });

  const result = ArrayStorage.zeros([realValues.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < realValues.length; i++) {
    resultData[i] = realValues[i]!;
  }
  return result;
}

/**
 * Trim leading and/or trailing zeros from a 1-D array.
 *
 * @param filt - Input 1-D array
 * @param trim - 'fb' to trim front and back, 'f' for front only, 'b' for back only (default: 'fb')
 * @returns Trimmed array
 */
export function trim_zeros(filt: ArrayStorage, trim: 'f' | 'b' | 'fb' = 'fb'): ArrayStorage {
  const dtype = filt.dtype;
  const data = filt.data;
  const size = filt.size;
  const off = filt.offset;

  if (size === 0) {
    return ArrayStorage.zeros([0], dtype);
  }

  let first = 0;
  let last = size - 1;

  if (isComplexDType(dtype)) {
    // Complex: check interleaved re/im pairs
    const cdata = data as Float64Array;
    if (trim !== 'b') {
      while (first < size && cdata[(off + first) * 2] === 0 && cdata[(off + first) * 2 + 1] === 0)
        first++;
    }
    if (trim !== 'f') {
      while (last >= first && cdata[(off + last) * 2] === 0 && cdata[(off + last) * 2 + 1] === 0)
        last--;
    }
    if (first > last) return ArrayStorage.zeros([0], dtype);
    const newSize = last - first + 1;
    const result = ArrayStorage.zeros([newSize], dtype);
    (result.data as Float64Array).set(
      cdata.subarray((off + first) * 2, (off + first + newSize) * 2)
    );
    return result;
  }

  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    // BigInt types: compare with 0n
    if (trim !== 'b') {
      while (first < size && data[off + first] === 0n) first++;
    }
    if (trim !== 'f') {
      while (last >= first && data[off + last] === 0n) last--;
    }
  } else {
    // All other typed arrays: inline zero check, no closure overhead
    // Use subarray so V8 sees a single typed array type in the inner loop
    const slice = (data as Float64Array).subarray(off, off + size);
    if (trim !== 'b') {
      while (first < size && slice[first] === 0) first++;
    }
    if (trim !== 'f') {
      while (last >= first && slice[last] === 0) last--;
    }
  }

  if (first > last) return ArrayStorage.zeros([0], dtype);

  const newSize = last - first + 1;
  const result = ArrayStorage.zeros([newSize], dtype);
  // Bulk copy using native TypedArray.set() instead of element-wise loop
  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    (result.data as BigInt64Array).set(data.subarray(off + first, off + first + newSize));
  } else {
    (result.data as Float64Array).set(
      (data as Float64Array).subarray(off + first, off + first + newSize)
    );
  }
  return result;
}

/**
 * Find the unique elements of an array, returning all optional outputs.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Object with values, indices, inverse_indices, and counts
 */
export function unique_all(x: ArrayStorage): {
  values: ArrayStorage;
  indices: ArrayStorage;
  inverse_indices: ArrayStorage;
  counts: ArrayStorage;
} {
  const result = unique(x, true, true, true);

  // unique with all flags returns an object
  const obj = result as {
    values: ArrayStorage;
    indices?: ArrayStorage;
    inverse?: ArrayStorage;
    counts?: ArrayStorage;
  };

  return {
    values: obj.values,
    indices: obj.indices!,
    inverse_indices: obj.inverse!,
    counts: obj.counts!,
  };
}

/**
 * Find the unique elements of an array and their counts.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Object with values and counts
 */
export function unique_counts(x: ArrayStorage): {
  values: ArrayStorage;
  counts: ArrayStorage;
} {
  const result = unique(x, false, false, true);

  // unique with returnCounts=true returns an object
  const obj = result as {
    values: ArrayStorage;
    counts?: ArrayStorage;
  };

  return {
    values: obj.values,
    counts: obj.counts!,
  };
}

/**
 * Find the unique elements of an array and their inverse indices.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Object with values and inverse_indices
 */
export function unique_inverse(x: ArrayStorage): {
  values: ArrayStorage;
  inverse_indices: ArrayStorage;
} {
  const result = unique(x, false, true, false);

  // unique with returnInverse=true returns an object
  const obj = result as {
    values: ArrayStorage;
    inverse?: ArrayStorage;
  };

  return {
    values: obj.values,
    inverse_indices: obj.inverse!,
  };
}

/**
 * Find the unique elements of an array (values only).
 *
 * This is equivalent to unique(x) but with a clearer name for the Array API.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Array of unique values, sorted
 */
export function unique_values(x: ArrayStorage): ArrayStorage {
  return unique(x) as ArrayStorage;
}
