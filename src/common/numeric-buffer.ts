/**
 * Helpers for JS loops over numeric arrays that must not index the array's own
 * typed array directly.
 *
 * A loop indexing the data is one load or store site shared by every dtype that
 * reaches it, and V8 falls back to its generic path once a site has seen more
 * than four typed-array classes, which makes those loops many times slower. These
 * helpers read and write through a Float64Array instead, converting with the
 * per-dtype loops in dtype-loops. Every non-bigint dtype round-trips through
 * float64 exactly, and complex arrays and 64-bit integers keep their own loops.
 */

import { type DType, isBigIntDType, isComplexDType, type TypedArray } from './dtype';
import { narrowFromF64, widenToF64 } from './internal/dtype-loops';
import type { ArrayStorage } from './storage';

export type NumericData = Exclude<TypedArray, BigInt64Array | BigUint64Array>;

/** True for dtypes whose elements are one plain number: everything but 64-bit integers and complex. */
export function isNumericDType(dtype: DType): boolean {
  return !isBigIntDType(dtype) && !isComplexDType(dtype);
}

/**
 * The first size values of a numeric array as a Float64Array: the data itself
 * for float64, otherwise a widened copy. Never mutate the result.
 */
export function asFloat64(data: NumericData, size: number): Float64Array {
  if (data instanceof Float64Array) return data;
  const out = new Float64Array(size);
  widenToF64(data, 0, size, out);
  return out;
}

/** Store float64 values into a numeric array, wrapping as an element store does. */
export function storeFloat64(target: NumericData, values: Float64Array): void {
  narrowFromF64(target, 0, values.length, values);
}

/**
 * Buffer for a loop that produces n values for a numeric array: the data itself
 * for float64, otherwise a Float64Array that commitFill stores afterwards.
 */
export function fillTarget(data: NumericData, n: number): Float64Array {
  return data instanceof Float64Array ? data : new Float64Array(n);
}

/** Store a fillTarget buffer into the data; a no-op when the buffer is the data. */
export function commitFill(data: NumericData, target: Float64Array): void {
  if (target !== data) storeFloat64(data, target);
}

/** Converting copy of the first size values, with the same wrapping as an element store. */
export function setConverted(target: NumericData, source: NumericData, size: number): void {
  narrowFromF64(target, 0, size, asFloat64(source, size));
}

/**
 * The elements of a numeric storage in row-major order as a Float64Array,
 * whatever its strides or offset. Never mutate the result: it can be the
 * storage's own data.
 */
export function logicalValues(storage: ArrayStorage): Float64Array {
  const size = storage.size;
  const data = storage.data as NumericData;
  if (storage.isCContiguous) {
    const view =
      storage.offset === 0 && data.length === size
        ? data
        : data.subarray(storage.offset, storage.offset + size);
    return asFloat64(view, size);
  }

  const source = asFloat64(data, data.length);
  const out = new Float64Array(size);
  const shape = storage.shape;
  const strides = storage.strides;
  const ndim = shape.length;
  const index = new Array<number>(ndim).fill(0);
  let position = storage.offset;
  for (let i = 0; i < size; i++) {
    out[i] = source[position]!;
    for (let d = ndim - 1; d >= 0; d--) {
      index[d]!++;
      position += strides[d]!;
      if (index[d]! < shape[d]!) break;
      position -= index[d]! * strides[d]!;
      index[d] = 0;
    }
  }
  return out;
}
