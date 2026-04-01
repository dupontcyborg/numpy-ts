/**
 * WASM-accelerated reduction nanmin.
 *
 * Reduction: result = nanmin(a[0..N])
 * Returns null if WASM can't handle this case.
 * Only float64 and float32.
 */

import { reduce_nanmin_f64, reduce_nanmin_f32 } from './bins/reduce_nanmin.wasm';
import { resetScratchAllocator, resolveInputPtr, f16InputToScratchF32 } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_nanmin_f64,
  float32: reduce_nanmin_f32,
  float16: reduce_nanmin_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
};

/**
 * WASM-accelerated reduction nanmin (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceNanmin(a: ArrayStorage): number | null {
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
