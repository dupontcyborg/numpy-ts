/**
 * WASM-accelerated reduction mean.
 *
 * Reduction: result = mean(a[0..N])
 * Returns null if WASM can't handle this case.
 * Unsigned types use SEPARATE kernels (for correct floatFromInt).
 */

import {
  reduce_mean_f64,
  reduce_mean_f32,
  reduce_mean_i64,
  reduce_mean_i32,
  reduce_mean_i16,
  reduce_mean_i8,
  reduce_mean_u64,
  reduce_mean_u32,
  reduce_mean_u16,
  reduce_mean_u8,
  reduce_mean_strided_f64,
  reduce_mean_strided_f32,
  reduce_mean_strided_i64,
  reduce_mean_strided_i32,
  reduce_mean_strided_i16,
  reduce_mean_strided_i8,
  reduce_mean_strided_u64,
  reduce_mean_strided_u32,
  reduce_mean_strided_u16,
  reduce_mean_strided_u8,
} from './bins/reduce_mean.wasm';
import {
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  wasmMalloc,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_mean_f64,
  float32: reduce_mean_f32,
  float16: reduce_mean_f32,
  int64: reduce_mean_i64,
  uint64: reduce_mean_u64,
  int32: reduce_mean_i32,
  uint32: reduce_mean_u32,
  int16: reduce_mean_i16,
  uint16: reduce_mean_u16,
  int8: reduce_mean_i8,
  uint8: reduce_mean_u8,
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
 * WASM-accelerated reduction mean (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceMean(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
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

// --- Strided axis reduction ---

type StridedFn = (aPtr: number, outPtr: number, outer: number, axis: number, inner: number) => void;

const stridedKernels: Partial<Record<DType, StridedFn>> = {
  float64: reduce_mean_strided_f64,
  float32: reduce_mean_strided_f32,
  float16: reduce_mean_strided_f32,
  int64: reduce_mean_strided_i64,
  uint64: reduce_mean_strided_u64,
  int32: reduce_mean_strided_i32,
  uint32: reduce_mean_strided_u32,
  int16: reduce_mean_strided_i16,
  uint16: reduce_mean_strided_u16,
  int8: reduce_mean_strided_i8,
  uint8: reduce_mean_strided_u8,
};

/**
 * WASM-accelerated strided mean along an axis.
 * Output is always float64. Returns output ArrayStorage, or null if WASM can't handle.
 */
export function wasmReduceMeanStrided(
  a: ArrayStorage,
  outerSize: number,
  axisSize: number,
  innerSize: number
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const totalSize = outerSize * axisSize * innerSize;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = stridedKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!kernel || !InCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBpe = Float64Array.BYTES_PER_ELEMENT;
  const outSize = outerSize * innerSize;

  const outRegion = wasmMalloc(outSize * outBpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let inPtr: number;
  if (dtype === 'float16') {
    inPtr = f16InputToScratchF32(a, totalSize);
  } else {
    inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, totalSize, inBpe);
  }

  kernel(inPtr, outRegion.ptr, outerSize, axisSize, innerSize);

  return ArrayStorage.fromWasmRegion(
    [outSize],
    'float64',
    outRegion,
    outSize,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
