/**
 * Set operations
 */

import { ArrayStorage } from '../core/storage';
import { type DType } from '../core/dtype';

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

  const unique1 = unique(ar1) as ArrayStorage;
  const unique2 = unique(ar2) as ArrayStorage;

  const set2 = new Set<number>();
  for (let i = 0; i < unique2.size; i++) {
    set2.add(Number(unique2.data[i]!));
  }

  const intersection: number[] = [];
  for (let i = 0; i < unique1.size; i++) {
    const val = Number(unique1.data[i]!);
    if (set2.has(val)) {
      intersection.push(val);
    }
  }

  intersection.sort((a, b) => a - b);

  const result = ArrayStorage.zeros([intersection.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < intersection.length; i++) {
    resultData[i] = intersection[i]!;
  }
  return result;
}

/**
 * Test whether each element of an ND array is also present in a second array
 */
export function isin(element: ArrayStorage, testElements: ArrayStorage): ArrayStorage {
  const shape = Array.from(element.shape);
  const size = element.size;

  const testSet = new Set<number>();
  for (let i = 0; i < testElements.size; i++) {
    testSet.add(Number(testElements.data[i]!));
  }

  const result = ArrayStorage.zeros(shape, 'bool');
  const resultData = result.data as Uint8Array;
  const elementData = element.data;

  for (let i = 0; i < size; i++) {
    const val = Number(elementData[i]!);
    resultData[i] = testSet.has(val) ? 1 : 0;
  }

  return result;
}

/**
 * Find the set difference of two arrays
 */
export function setdiff1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;

  const unique1 = unique(ar1) as ArrayStorage;

  const set2 = new Set<number>();
  for (let i = 0; i < ar2.size; i++) {
    set2.add(Number(ar2.data[i]!));
  }

  const diff: number[] = [];
  for (let i = 0; i < unique1.size; i++) {
    const val = Number(unique1.data[i]!);
    if (!set2.has(val)) {
      diff.push(val);
    }
  }

  const result = ArrayStorage.zeros([diff.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < diff.length; i++) {
    resultData[i] = diff[i]!;
  }
  return result;
}

/**
 * Find the set exclusive-or of two arrays
 */
export function setxor1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;

  const unique1 = unique(ar1) as ArrayStorage;
  const unique2 = unique(ar2) as ArrayStorage;

  const set1 = new Set<number>();
  const set2 = new Set<number>();

  for (let i = 0; i < unique1.size; i++) {
    set1.add(Number(unique1.data[i]!));
  }
  for (let i = 0; i < unique2.size; i++) {
    set2.add(Number(unique2.data[i]!));
  }

  const xor: number[] = [];
  for (const val of set1) {
    if (!set2.has(val)) {
      xor.push(val);
    }
  }
  for (const val of set2) {
    if (!set1.has(val)) {
      xor.push(val);
    }
  }

  xor.sort((a, b) => a - b);

  const result = ArrayStorage.zeros([xor.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < xor.length; i++) {
    resultData[i] = xor[i]!;
  }
  return result;
}

/**
 * Find the union of two arrays
 */
export function union1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const dtype = ar1.dtype;

  const unique1 = unique(ar1) as ArrayStorage;
  const unique2 = unique(ar2) as ArrayStorage;

  const unionSet = new Set<number>();

  for (let i = 0; i < unique1.size; i++) {
    unionSet.add(Number(unique1.data[i]!));
  }
  for (let i = 0; i < unique2.size; i++) {
    unionSet.add(Number(unique2.data[i]!));
  }

  const unionArr = Array.from(unionSet);
  unionArr.sort((a, b) => a - b);

  const result = ArrayStorage.zeros([unionArr.length], dtype as DType);
  const resultData = result.data;
  for (let i = 0; i < unionArr.length; i++) {
    resultData[i] = unionArr[i]!;
  }
  return result;
}
