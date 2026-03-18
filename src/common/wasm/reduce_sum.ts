/**
 * WASM-accelerated reduction sum.
 *
 * Reduction: result = sum(a[0..N])
 * Returns null if WASM can't handle this case.
 * uint types route to signed kernels (wrapping addition gives same bits).
 */

import {
  reduce_sum_f64,
  reduce_sum_f32,
  reduce_sum_i64,
  reduce_sum_i32,
  reduce_sum_strided_f64,
  reduce_sum_strided_f32,
  reduce_sum_strided_i64,
  reduce_sum_strided_i32,
  reduce_sum_strided_i16,
  reduce_sum_strided_i8,
  reduce_sum_strided_u64,
  reduce_sum_strided_u32,
  reduce_sum_strided_u16,
  reduce_sum_strided_u8,
} from './bins/reduce_sum.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_sum_f64,
  float32: reduce_sum_f32,
  int64: reduce_sum_i64,
  uint64: reduce_sum_i64,
  int32: reduce_sum_i32,
  uint32: reduce_sum_i32,
  // int8/int16/uint8/uint16 excluded: WASM accumulates in native type which
  // overflows for large arrays. JS fallback promotes to int64 (matches NumPy).
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

/**
 * WASM-accelerated reduction sum (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceSum(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  ensureMemory(size * bpe);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
  const aPtr = copyIn(aData);

  return Number(kernel(aPtr, size));
}

// --- Strided axis reduction (all output f64 to avoid overflow and BigInt overhead) ---

type StridedFn = (aPtr: number, outPtr: number, outer: number, axis: number, inner: number) => void;

const stridedKernels: Partial<Record<DType, StridedFn>> = {
  float64: reduce_sum_strided_f64,
  float32: reduce_sum_strided_f32,
  int64: reduce_sum_strided_i64,
  uint64: reduce_sum_strided_u64,
  int32: reduce_sum_strided_i32,
  uint32: reduce_sum_strided_u32,
  int16: reduce_sum_strided_i16,
  uint16: reduce_sum_strided_u16,
  int8: reduce_sum_strided_i8,
  uint8: reduce_sum_strided_u8,
};

/**
 * WASM-accelerated strided sum along an axis.
 * Output is always f64 (the JS caller converts to the correct output dtype).
 * Returns f64 ArrayStorage, or null if WASM can't handle.
 */
export function wasmReduceSumStrided(
  a: ArrayStorage,
  outerSize: number,
  axisSize: number,
  innerSize: number
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const totalSize = outerSize * axisSize * innerSize;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = stridedKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!kernel || !InCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outSize = outerSize * innerSize;

  ensureMemory(totalSize * inBpe + outSize * 8);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + totalSize) as TypedArray;
  const inPtr = copyIn(aData);
  const outPtr = alloc(outSize * 8);

  kernel(inPtr, outPtr, outerSize, axisSize, innerSize);

  const outData = copyOut(
    outPtr,
    outSize,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, [outSize], 'float64');
}
