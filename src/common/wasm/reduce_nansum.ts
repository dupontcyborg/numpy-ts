/**
 * WASM-accelerated reduction nansum.
 *
 * Reduction: result = nansum(a[0..N])
 * Returns null if WASM can't handle this case.
 * Only float64 and float32 — TS routes int/uint to regular sum.
 */

import { reduce_nansum_f64, reduce_nansum_f32 } from './bins/reduce_nansum.wasm';
import { resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_nansum_f64,
  float32: reduce_nansum_f32,
  // float16 excluded: must accumulate in f16 precision to match NumPy overflow behavior
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
};

/**
 * WASM-accelerated reduction nansum (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceNansum(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  return Number(kernel(aPtr, size));
}
