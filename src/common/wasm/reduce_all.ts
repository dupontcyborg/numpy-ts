/**
 * WASM-accelerated reduction all (logical AND).
 *
 * Reduction: result = 1 if 0 if any a[i] == 0, else 1.
 * Returns null if WASM can't handle this case.
 * uint types route to signed kernels (non-zero check is sign-agnostic).
 */

import {
  reduce_all_f64,
  reduce_all_f32,
  reduce_all_i64,
  reduce_all_i32,
  reduce_all_i16,
  reduce_all_i8,
} from './bins/reduce_all.wasm';
import { resetScratchAllocator, resolveInputPtr, f16InputToScratchF32 } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_all_f64,
  float32: reduce_all_f32,
  float16: reduce_all_f32,
  int64: reduce_all_i64,
  uint64: reduce_all_i64,
  int32: reduce_all_i32,
  uint32: reduce_all_i32,
  int16: reduce_all_i16,
  uint16: reduce_all_i16,
  int8: reduce_all_i8,
  uint8: reduce_all_i8,
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
 * WASM-accelerated reduction all (no axis, full array).
 * Returns 0 or 1 as number, or null if WASM can't handle.
 */
export function wasmReduceAll(a: ArrayStorage): number | null {
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

  return kernel(aPtr, size);
}
