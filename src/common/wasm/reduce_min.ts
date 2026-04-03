/**
 * WASM-accelerated reduction min.
 *
 * Reduction: result = min(a[0..N])
 * Returns null if WASM can't handle this case.
 * Unsigned types use SEPARATE kernels.
 */

import {
  reduce_min_f64,
  reduce_min_f32,
  reduce_min_i64,
  reduce_min_i32,
  reduce_min_i16,
  reduce_min_i8,
  reduce_min_u64,
  reduce_min_u32,
  reduce_min_u16,
  reduce_min_u8,
  reduce_min_strided_f64,
  reduce_min_strided_f32,
  reduce_min_strided_i64,
  reduce_min_strided_i32,
  reduce_min_strided_i16,
  reduce_min_strided_i8,
  reduce_min_strided_u64,
  reduce_min_strided_u32,
  reduce_min_strided_u16,
  reduce_min_strided_u8,
} from './bins/reduce_min.wasm';
import {
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  wasmMalloc,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_min_f64,
  float32: reduce_min_f32,
  float16: reduce_min_f32,
  int64: reduce_min_i64,
  uint64: reduce_min_u64,
  int32: reduce_min_i32,
  uint32: reduce_min_u32,
  int16: reduce_min_i16,
  uint16: reduce_min_u16,
  int8: reduce_min_i8,
  uint8: reduce_min_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
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
 * WASM-accelerated reduction min (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceMin(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let aPtr: number;
  if (dtype === 'float16') {
    aPtr = f16InputToScratchF32(a, size);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  }

  return Number(kernel(aPtr, size));
}

// --- Strided axis reduction (output dtype matches input dtype) ---

type StridedFn = (aPtr: number, outPtr: number, outer: number, axis: number, inner: number) => void;

const stridedKernels: Partial<Record<DType, StridedFn>> = {
  float64: reduce_min_strided_f64,
  float32: reduce_min_strided_f32,
  float16: reduce_min_strided_f32,
  int64: reduce_min_strided_i64,
  uint64: reduce_min_strided_u64,
  int32: reduce_min_strided_i32,
  uint32: reduce_min_strided_u32,
  int16: reduce_min_strided_i16,
  uint16: reduce_min_strided_u16,
  int8: reduce_min_strided_i8,
  uint8: reduce_min_strided_u8,
};

// Output Ctor matches input dtype (min doesn't promote)
const outCtorMap: Partial<
  Record<DType, new (buf: ArrayBuffer, off: number, len: number) => TypedArray>
> = {
  float32: Float32Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  float16: Float32Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int64: BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint64: BigUint64Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int32: Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint32: Uint32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int16: Int16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint16: Uint16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int8: Int8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint8: Uint8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
};

/**
 * WASM-accelerated strided min along an axis.
 * Output dtype matches input dtype. Returns output ArrayStorage, or null if WASM can't handle.
 */
export function wasmReduceMinStrided(
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
  const OutCtor = outCtorMap[dtype];
  if (!kernel || !InCtor || !OutCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBpe = inBpe; // output type matches input type for min
  const outSize = outerSize * innerSize;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let inPtr: number;
  if (dtype === 'float16') {
    inPtr = f16InputToScratchF32(a, totalSize);
  } else {
    inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, totalSize, inBpe);
  }

  // f16: output as f32, then convert to f16 via wasmMalloc region
  if (dtype === 'float16') {
    const outRegion = wasmMalloc(outSize * 4);
    if (!outRegion) return null;
    kernel(inPtr, outRegion.ptr, outerSize, axisSize, innerSize);
    const f16Region = f32OutputToF16Region(outRegion, outSize);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [outSize],
      dtype,
      f16Region,
      outSize,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const outRegion2 = wasmMalloc(outSize * outBpe);
  if (!outRegion2) return null;
  kernel(inPtr, outRegion2.ptr, outerSize, axisSize, innerSize);

  return ArrayStorage.fromWasmRegion([outSize], dtype, outRegion2, outSize, OutCtor);
}
