/**
 * WASM-accelerated reduction var.
 *
 * Reduction: result = var(a[0..N])
 * Returns null if WASM can't handle this case.
 * Unsigned types use SEPARATE kernels.
 */

import {
  reduce_var_f64,
  reduce_var_f32,
  reduce_var_i64,
  reduce_var_i32,
  reduce_var_i16,
  reduce_var_i8,
  reduce_var_u64,
  reduce_var_u32,
  reduce_var_u16,
  reduce_var_u8,
} from './bins/reduce_var.wasm';
import { resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_var_f64,
  float32: reduce_var_f32,
  int64: reduce_var_i64,
  uint64: reduce_var_u64,
  int32: reduce_var_i32,
  uint32: reduce_var_u32,
  int16: reduce_var_i16,
  uint16: reduce_var_u16,
  int8: reduce_var_i8,
  uint8: reduce_var_u8,
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
 * WASM-accelerated reduction var (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceVar(a: ArrayStorage): number | null {
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
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  return Number(kernel(aPtr, size));
}
