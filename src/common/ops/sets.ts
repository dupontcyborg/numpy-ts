/**
 * Set operations
 */

import { type DType, isComplexDType, isFloatDType, promoteDTypes, type TypedArray } from '../dtype';
import { widenToF64 } from '../internal/dtype-loops';
import { ArrayStorage } from '../storage';
import { castStorage, concatenate, ravel } from './shape';

// Helper: compare complex numbers lexicographically
function complexCompare(aRe: number, aIm: number, bRe: number, bIm: number): number {
  const aIsNaN = Number.isNaN(aRe) || Number.isNaN(aIm);
  const bIsNaN = Number.isNaN(bRe) || Number.isNaN(bIm);
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
  const aIsNaN = Number.isNaN(aRe) || Number.isNaN(aIm);
  const bIsNaN = Number.isNaN(bRe) || Number.isNaN(bIm);
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
  returnCounts: boolean,
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
 * Whether every value in a 64-bit integer range is exactly representable as a
 * double, so the generic Float64Array path below cannot merge distinct values.
 * The exact BigInt path this gates is correct for the whole 64-bit range but
 * sorts via a JS comparator that loads BigInts, far costlier than a numeric
 * sort; real int64 data almost always fits in the safe range, so this one
 * linear scan buys the fast path most of the time while keeping the exact
 * path for values that need it. The boundary check is deliberately
 * conservative (`|v| < 2^53` rather than `<= 2^53`), which at worst takes the
 * exact path one value early.
 */
function allExactInF64(src: BigInt64Array | BigUint64Array, off: number, size: number): boolean {
  const LIMIT = 9007199254740992n; // 2^53
  for (let i = 0; i < size; i++) {
    const v = src[off + i]!;
    if (v >= LIMIT || v <= -LIMIT) return false;
  }
  return true;
}

/**
 * Find the unique elements of an array
 */
export function unique(
  a: ArrayStorage,
  returnIndex: boolean = false,
  returnInverse: boolean = false,
  returnCounts: boolean = false,
  axis?: number,
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
    const normalizedAxis = axis < 0 ? ndim + axis : axis;
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
    let lastKey: string | undefined;
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

    let lastRe: number | undefined;
    let lastIm: number | undefined;
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
      if (Number.isNaN(re) || Number.isNaN(im)) {
        nanIdx = i;
      } else {
        valueToUniqueIdx.set(`${re},${im}`, i);
      }
    }
    for (let i = 0; i < size; i++) {
      const re = complexData[(off + i) * 2]!;
      const im = complexData[(off + i) * 2 + 1]!;
      if (Number.isNaN(re) || Number.isNaN(im)) {
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
      returnCounts,
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
        returnCounts,
      );
    }
  }

  // int64/uint64: sort and compare the BigInts directly. Every path below funnels
  // values through a Float64Array, which merges distinct 64-bit values above 2^53
  // — unique([2^53, 2^53+1, 2^53+2, 2^53+3]) came back with three elements, and
  // the wrong ones.
  if (
    (data instanceof BigInt64Array || data instanceof BigUint64Array) &&
    !allExactInF64(data as BigInt64Array | BigUint64Array, off, size)
  ) {
    const src = data as BigInt64Array | BigUint64Array;
    const at = (i: number): bigint => src[off + i]!;

    const order = new Array<number>(size);
    for (let i = 0; i < size; i++) order[i] = i;
    // Tie-break on the original index so the first occurrence of each value wins,
    // which is what return_index reports.
    order.sort((x, y) => {
      const av = at(x);
      const bv = at(y);
      return av < bv ? -1 : av > bv ? 1 : x - y;
    });

    const uniqVals: bigint[] = [];
    const firstIdx: number[] = [];
    const counts: number[] = [];
    for (const idx of order) {
      const v = at(idx);
      if (uniqVals.length === 0 || v !== uniqVals[uniqVals.length - 1]) {
        uniqVals.push(v);
        firstIdx.push(idx);
        counts.push(1);
      } else {
        counts[counts.length - 1]! += 1;
      }
    }

    const values = ArrayStorage.zeros([uniqVals.length], dtype as DType);
    const vd = values.data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < uniqVals.length; i++) vd[i] = uniqVals[i]!;

    if (!returnIndex && !returnInverse && !returnCounts) return values;

    const out: {
      values: ArrayStorage;
      indices?: ArrayStorage;
      inverse?: ArrayStorage;
      counts?: ArrayStorage;
    } = { values };

    if (returnIndex) {
      const ind = ArrayStorage.zeros([firstIdx.length], 'int32');
      const id = ind.data as Int32Array;
      for (let i = 0; i < firstIdx.length; i++) id[i] = firstIdx[i]!;
      out.indices = ind;
    }
    if (returnInverse) {
      const pos = new Map<bigint, number>();
      for (let i = 0; i < uniqVals.length; i++) pos.set(uniqVals[i]!, i);
      const inv = ArrayStorage.zeros([size], 'int32');
      const ivd = inv.data as Int32Array;
      for (let i = 0; i < size; i++) ivd[i] = pos.get(at(i))!;
      out.inverse = inv;
    }
    if (returnCounts) {
      const cnt = ArrayStorage.zeros([counts.length], 'int32');
      const cd = cnt.data as Int32Array;
      for (let i = 0; i < counts.length; i++) cd[i] = counts[i]!;
      out.counts = cnt;
    }
    return out;
  }

  const isBigInt = data instanceof BigInt64Array || data instanceof BigUint64Array;
  const isFloat = dtype === 'float64' || dtype === 'float32' || dtype === 'float16';

  // Fast path: when we don't need original indices, sort values directly
  if (!returnIndex && !returnInverse) {
    const vals = new Float64Array(size);
    widenToF64(data, off, size, vals);
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
  // Dispatched per dtype; `Number(data[off + i])` read through the TypedArray
  // union, which goes megamorphic across a dtype sweep.
  widenToF64(data, off, size, vals);
  for (let i = 0; i < size; i++) sortedIdxs[i] = i;

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

  // The dedup walk already visits every element in sorted order and knows which
  // unique value it belongs to, so record the inverse here. Rebuilding it
  // afterwards cost a Map<number, number> plus one float-keyed lookup per
  // element — for no information the walk did not already have.
  const inverseDirect = returnInverse ? new Int32Array(size) : null;

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
    if (inverseDirect) inverseDirect[idx] = uniqueValues.length - 1;
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
    const inverseResult = ArrayStorage.zeros([size], 'int32');
    (inverseResult.data as Int32Array).set(inverseDirect!);
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

// ─── Membership and set algebra on flattened inputs ──────────────────────────
// These follow NumPy's algorithms step for step, so that the assume_unique flag on
// non-unique input produces the same (unspecified) results NumPy does.

function flatten1d(a: ArrayStorage): ArrayStorage {
  return ravel(a);
}

/** Copy the elements at the given positions of a contiguous 1-D storage. */
function gather(src: ArrayStorage, picks: ArrayLike<number>): ArrayStorage {
  const out = ArrayStorage.zeros([picks.length], src.dtype);
  const from = src.data as unknown as { [i: number]: number | bigint };
  const to = out.data as unknown as { [i: number]: number | bigint };
  const off = src.offset;
  if (isComplexDType(src.dtype)) {
    for (let i = 0; i < picks.length; i++) {
      const at = (off + picks[i]!) * 2;
      to[i * 2] = from[at]!;
      to[i * 2 + 1] = from[at + 1]!;
    }
  } else {
    for (let i = 0; i < picks.length; i++) to[i] = from[off + picks[i]!]!;
  }
  return out;
}

/** Equality of two positions of a contiguous 1-D storage; NaN never equals anything. */
function positionEquals(s: ArrayStorage): (i: number, j: number) => boolean {
  const d = s.data as unknown as { [i: number]: number | bigint };
  const off = s.offset;
  if (isComplexDType(s.dtype)) {
    return (i, j) => {
      const a = (off + i) * 2;
      const b = (off + j) * 2;
      return d[a] === d[b] && d[a + 1] === d[b + 1];
    };
  }
  return (i, j) => d[off + i] === d[off + j];
}

/** Stable ordering of a contiguous 1-D storage, NaN last and complex lexicographic. */
function stableOrder(s: ArrayStorage): number[] {
  const n = s.size;
  const order = Array.from({ length: n }, (_, i) => i);
  const off = s.offset;
  if (isComplexDType(s.dtype)) {
    const d = s.data as Float64Array | Float32Array;
    return order.sort((a, b) =>
      complexCompare(
        d[(off + a) * 2]!,
        d[(off + a) * 2 + 1]!,
        d[(off + b) * 2]!,
        d[(off + b) * 2 + 1]!,
      ),
    );
  }
  const d = s.data as unknown as { [i: number]: number | bigint };
  return order.sort((a, b) => {
    const x = d[off + a]!;
    const y = d[off + b]!;
    if (x < y) return -1;
    if (x > y) return 1;
    if (x === y) return 0;
    // Only NaN reaches here: it sorts after everything else.
    const xNaN = x !== x;
    const yNaN = y !== y;
    return xNaN === yNaN ? 0 : xNaN ? 1 : -1;
  });
}

function isIntOrBool(dtype: DType): boolean {
  return dtype === 'bool' || (!isComplexDType(dtype) && !isFloatDType(dtype));
}

// Largest value of each integer dtype, as NumPy's table method bounds the span of the second
// array by it (bool is widened to uint8 first).
const INT_MAX: Partial<Record<DType, bigint>> = {
  bool: 255n,
  uint8: 255n,
  uint16: 65535n,
  uint32: 4294967295n,
  uint64: 18446744073709551615n,
  int8: 127n,
  int16: 32767n,
  int32: 2147483647n,
  int64: 9223372036854775807n,
};

/**
 * Whether NumPy would answer membership with the sort-and-compare-neighbours method. That
 * method is only exact for duplicate-free input, which is why it matters under assume_unique;
 * the table method (small-range integers) and the loop method (few test values) are exact for
 * any input.
 */
function usesSortMethod(a1: ArrayStorage, a2: ArrayStorage): boolean {
  const n1 = a1.size;
  const n2 = a2.size;
  if (isIntOrBool(a1.dtype) && isIntOrBool(a2.dtype)) {
    if (n2 === 0) return false;
    const d = a2.data as unknown as { [i: number]: number | bigint };
    let lo = BigInt(d[a2.offset]!);
    let hi = lo;
    for (let i = 1; i < n2; i++) {
      const v = BigInt(d[a2.offset + i]!);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const range = hi - lo;
    if (range <= BigInt(6 * (n1 + n2)) && range <= INT_MAX[a2.dtype]!) return false;
  }
  return !(n2 < 10 * n1 ** 0.145);
}

/** Positions of a1 whose value occurs in a2 (a1 and a2 flat and of one dtype), NaN never matching. */
function exactMembership(a1: ArrayStorage, a2: ArrayStorage): Uint8Array {
  const out = new Uint8Array(a1.size);
  const d1 = a1.data as unknown as { [i: number]: number | bigint };
  const d2 = a2.data as unknown as { [i: number]: number | bigint };
  if (isComplexDType(a1.dtype)) {
    const seen = new Set<string>();
    for (let i = 0; i < a2.size; i++) {
      const re = d2[(a2.offset + i) * 2] as number;
      const im = d2[(a2.offset + i) * 2 + 1] as number;
      if (re === re && im === im) seen.add(`${re},${im}`);
    }
    for (let i = 0; i < a1.size; i++) {
      const re = d1[(a1.offset + i) * 2] as number;
      const im = d1[(a1.offset + i) * 2 + 1] as number;
      if (re === re && im === im && seen.has(`${re},${im}`)) out[i] = 1;
    }
    return out;
  }
  // A Set matches NaN with NaN, which equality never does, so NaN is kept out of it.
  const seen = new Set<number | bigint>();
  for (let i = 0; i < a2.size; i++) {
    const v = d2[a2.offset + i]!;
    if (v === v) seen.add(v);
  }
  for (let i = 0; i < a1.size; i++) if (seen.has(d1[a1.offset + i]!)) out[i] = 1;
  return out;
}

/** Whether each element of the flat a1 occurs in the flat a2, following NumPy's method choice. */
function membership(
  a1: ArrayStorage,
  a2: ArrayStorage,
  assumeUnique: boolean,
  invert: boolean,
): Uint8Array {
  const dtype = promoteDTypes(a1.dtype, a2.dtype);
  const c1 = castStorage(a1, dtype);
  const c2 = castStorage(a2, dtype);
  let found: Uint8Array;
  if (assumeUnique && usesSortMethod(a1, a2)) {
    const joined = concatenate([c1, c2], 0);
    const order = stableOrder(joined);
    const eq = positionEquals(joined);
    found = new Uint8Array(a1.size);
    for (let k = 0; k < order.length - 1; k++) {
      const at = order[k]!;
      if (at < a1.size && eq(at, order[k + 1]!)) found[at] = 1;
    }
    joined.dispose();
  } else {
    found = exactMembership(c1, c2);
  }
  if (c1 !== a1) c1.dispose();
  if (c2 !== a2) c2.dispose();
  if (invert) for (let i = 0; i < found.length; i++) found[i] = found[i]! ^ 1;
  return found;
}

function boolStorage(mask: Uint8Array, shape: number[]): ArrayStorage {
  const result = ArrayStorage.zeros(shape, 'bool');
  (result.data as Uint8Array).set(mask);
  return result;
}

function uniqueFlat(a: ArrayStorage): ArrayStorage {
  const flat = flatten1d(a);
  const result = unique(flat) as ArrayStorage;
  flat.dispose();
  return result;
}

/**
 * Test whether each element of a 1-D array is also present in a second array. The result is
 * 1-D even for multi-dimensional input. With assume_unique the inputs are trusted to hold no
 * duplicates, which on duplicated input gives the same unreliable answers NumPy does.
 * @deprecated Use {@link isin} instead. `in1d` follows NumPy's deprecation and will be removed in a future release.
 */
export function in1d(
  ar1: ArrayStorage,
  ar2: ArrayStorage,
  assumeUnique: boolean = false,
  invert: boolean = false,
): ArrayStorage {
  const a1 = flatten1d(ar1);
  const a2 = flatten1d(ar2);
  const mask = membership(a1, a2, assumeUnique, invert);
  return boolStorage(mask, [ar1.size]);
}

/**
 * Test whether each element of an ND array is also present in a second array, keeping the
 * shape of the first. NaN is never a member, so with invert it is always reported as absent.
 */
export function isin(
  element: ArrayStorage,
  testElements: ArrayStorage,
  assumeUnique: boolean = false,
  invert: boolean = false,
): ArrayStorage {
  const a1 = flatten1d(element);
  const a2 = flatten1d(testElements);
  const mask = membership(a1, a2, assumeUnique, invert);
  return boolStorage(mask, Array.from(element.shape));
}

/**
 * Find the intersection of two arrays, as the sorted common values. With returnIndices the
 * result is [values, indices into ar1, indices into ar2], each the first occurrence in the
 * flattened input.
 */
export function intersect1d(
  ar1: ArrayStorage,
  ar2: ArrayStorage,
  assumeUnique: boolean,
  returnIndices: true,
): [ArrayStorage, ArrayStorage, ArrayStorage];
export function intersect1d(
  ar1: ArrayStorage,
  ar2: ArrayStorage,
  assumeUnique?: boolean,
  returnIndices?: boolean,
): ArrayStorage;
export function intersect1d(
  ar1: ArrayStorage,
  ar2: ArrayStorage,
  assumeUnique: boolean = false,
  returnIndices: boolean = false,
): ArrayStorage | [ArrayStorage, ArrayStorage, ArrayStorage] {
  let a1: ArrayStorage;
  let a2: ArrayStorage;
  let first1: ArrayStorage | undefined;
  let first2: ArrayStorage | undefined;
  if (assumeUnique) {
    a1 = flatten1d(ar1);
    a2 = flatten1d(ar2);
  } else if (returnIndices) {
    const f1 = flatten1d(ar1);
    const f2 = flatten1d(ar2);
    const u1 = unique(f1, true) as { values: ArrayStorage; indices: ArrayStorage };
    const u2 = unique(f2, true) as { values: ArrayStorage; indices: ArrayStorage };
    a1 = u1.values;
    a2 = u2.values;
    first1 = u1.indices;
    first2 = u2.indices;
  } else {
    a1 = uniqueFlat(ar1);
    a2 = uniqueFlat(ar2);
  }

  const joined = concatenate([a1, a2], 0);
  const order = stableOrder(joined);
  const sorted = gather(joined, order);
  const eq = positionEquals(sorted);
  const picks: number[] = [];
  for (let k = 0; k < order.length - 1; k++) if (eq(k, k + 1)) picks.push(k);
  const common = gather(sorted, picks);

  let result: ArrayStorage | [ArrayStorage, ArrayStorage, ArrayStorage] = common;
  if (returnIndices) {
    const idx1 = ArrayStorage.zeros([picks.length], 'int32');
    const idx2 = ArrayStorage.zeros([picks.length], 'int32');
    const d1 = idx1.data as Int32Array;
    const d2 = idx2.data as Int32Array;
    for (let i = 0; i < picks.length; i++) {
      const in1 = order[picks[i]!]!;
      const in2 = order[picks[i]! + 1]! - a1.size;
      d1[i] = first1 ? (first1.data as Int32Array)[first1.offset + in1]! : in1;
      d2[i] = first2 ? (first2.data as Int32Array)[first2.offset + in2]! : in2;
    }
    result = [common, idx1, idx2];
  }

  joined.dispose();
  sorted.dispose();
  if (!assumeUnique) {
    a1.dispose();
    a2.dispose();
    first1?.dispose();
    first2?.dispose();
  }
  return result;
}

/**
 * Find the set difference of two arrays: the values of ar1 that are not in ar2. The result is
 * sorted unless assumeUnique is set, in which case ar1 is only flattened and keeps its order.
 */
export function setdiff1d(
  ar1: ArrayStorage,
  ar2: ArrayStorage,
  assumeUnique: boolean = false,
): ArrayStorage {
  const a1 = assumeUnique ? flatten1d(ar1) : uniqueFlat(ar1);
  const a2 = assumeUnique ? flatten1d(ar2) : uniqueFlat(ar2);
  const absent = membership(a1, a2, true, true);
  const picks: number[] = [];
  for (let i = 0; i < absent.length; i++) if (absent[i]) picks.push(i);
  const result = gather(a1, picks);
  if (!assumeUnique) {
    a1.dispose();
    a2.dispose();
  }
  return result;
}

/**
 * Find the set exclusive-or of two arrays: the sorted values present in exactly one of them.
 * NaN compares unequal to itself, so every NaN is kept.
 */
export function setxor1d(
  ar1: ArrayStorage,
  ar2: ArrayStorage,
  assumeUnique: boolean = false,
): ArrayStorage {
  const a1 = assumeUnique ? flatten1d(ar1) : uniqueFlat(ar1);
  const a2 = assumeUnique ? flatten1d(ar2) : uniqueFlat(ar2);
  const joined = concatenate([a1, a2], 0);
  if (!assumeUnique) {
    a1.dispose();
    a2.dispose();
  }
  if (joined.size === 0) return joined;

  const sorted = gather(joined, stableOrder(joined));
  joined.dispose();
  const eq = positionEquals(sorted);
  const picks: number[] = [];
  for (let i = 0; i < sorted.size; i++) {
    const differsFromPrev = i === 0 || !eq(i, i - 1);
    const differsFromNext = i === sorted.size - 1 || !eq(i, i + 1);
    if (differsFromPrev && differsFromNext) picks.push(i);
  }
  const result = gather(sorted, picks);
  sorted.dispose();
  return result;
}

/**
 * Find the union of two arrays: the sorted unique values of both, in their promoted dtype.
 */
export function union1d(ar1: ArrayStorage, ar2: ArrayStorage): ArrayStorage {
  const joined = concatenate([flatten1d(ar1), flatten1d(ar2)], 0);
  const result = unique(joined) as ArrayStorage;
  joined.dispose();
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
      cdata.subarray((off + first) * 2, (off + first + newSize) * 2),
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
      (data as Float64Array).subarray(off + first, off + first + newSize),
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
 * Find the unique elements of an array (values only) — equivalent to
 * unique(x), with a clearer name for the Array API.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Array of unique values, sorted
 */
export function unique_values(x: ArrayStorage): ArrayStorage {
  return unique(x) as ArrayStorage;
}
