/**
 * Set operations
 */

import { ArrayStorage } from '../core/storage';
import { isComplexDType, type DType } from '../core/dtype';

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
 * Find the unique elements of an array
 */
export function unique(
  a: ArrayStorage,
  returnIndex: boolean = false,
  returnInverse: boolean = false,
  returnCounts: boolean = false
):
  | ArrayStorage
  | {
      values: ArrayStorage;
      indices?: ArrayStorage;
      inverse?: ArrayStorage;
      counts?: ArrayStorage;
    } {
  const dtype = a.dtype;
  const size = a.size;
  const data = a.data;

  // Complex unique with lexicographic ordering
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    // Collect complex values with original indices
    const values: { re: number; im: number; index: number }[] = [];
    for (let i = 0; i < size; i++) {
      values.push({
        re: complexData[i * 2]!,
        im: complexData[i * 2 + 1]!,
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
      const isDifferent =
        lastRe === undefined || !complexEqual(re, im, lastRe!, lastIm!);

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
      const re = complexData[i * 2]!;
      const im = complexData[i * 2 + 1]!;
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

  // Collect values with original indices
  const values: { value: number; index: number }[] = [];
  for (let i = 0; i < size; i++) {
    values.push({ value: Number(data[i]!), index: i });
  }

  // Sort by value
  values.sort((x, y) => {
    if (isNaN(x.value) && isNaN(y.value)) return 0;
    if (isNaN(x.value)) return 1;
    if (isNaN(y.value)) return -1;
    return x.value - y.value;
  });

  // Find unique values
  const uniqueValues: number[] = [];
  const indices: number[] = [];
  const inverse: number[] = new Array(size);
  const counts: number[] = [];

  let lastValue: number | undefined = undefined;
  let currentCount = 0;

  for (let i = 0; i < values.length; i++) {
    const { value, index } = values[i]!;
    const isDifferent =
      lastValue === undefined ||
      (isNaN(value) && !isNaN(lastValue!)) ||
      (!isNaN(value) && isNaN(lastValue!)) ||
      (!isNaN(value) && !isNaN(lastValue!) && value !== lastValue);

    if (isDifferent) {
      if (lastValue !== undefined) {
        counts.push(currentCount);
      }
      uniqueValues.push(value);
      indices.push(index);
      currentCount = 1;
      lastValue = value;
    } else {
      currentCount++;
    }
  }
  if (currentCount > 0) {
    counts.push(currentCount);
  }

  // Build inverse mapping
  const valueToUniqueIdx = new Map<number, number>();
  let nanIdx = -1;
  for (let i = 0; i < uniqueValues.length; i++) {
    const v = uniqueValues[i]!;
    if (isNaN(v)) {
      nanIdx = i;
    } else {
      valueToUniqueIdx.set(v, i);
    }
  }
  for (let i = 0; i < size; i++) {
    const val = Number(data[i]!);
    if (isNaN(val)) {
      inverse[i] = nanIdx;
    } else {
      inverse[i] = valueToUniqueIdx.get(val)!;
    }
  }

  // Create result arrays
  const uniqueResult = ArrayStorage.zeros([uniqueValues.length], dtype as DType);
  const uniqueData = uniqueResult.data;
  for (let i = 0; i < uniqueValues.length; i++) {
    uniqueData[i] = uniqueValues[i]!;
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

// Helper: convert storage element to key string
function elementToKey(data: ArrayLike<number | bigint>, index: number, isComplex: boolean): string {
  if (isComplex) {
    const re = Number((data as Float64Array)[index * 2]);
    const im = Number((data as Float64Array)[index * 2 + 1]);
    return `${re},${im}`;
  }
  return String(Number(data[index]!));
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
    testSet.add(elementToKey(testElements.data, i, isComplex));
  }

  const result = ArrayStorage.zeros(shape, 'bool');
  const resultData = result.data as Uint8Array;

  for (let i = 0; i < size; i++) {
    const key = elementToKey(element.data, i, isComplex);
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
    set2.add(elementToKey(ar2.data, i, isComplex));
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

  const totalLen = xorIndices1.length + xorIndices2.length;

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
