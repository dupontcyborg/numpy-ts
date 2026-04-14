/**
 * WASM-accelerated isnan: out[i] = 1 if a[i] is NaN, else 0.
 * Returns null if WASM can't handle this case.
 */

import { isnan_f64, isnan_f32, isnan_u16 } from './bins/isnan.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type IsnanFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, IsnanFn>> = {
  float64: isnan_f64,
  float32: isnan_f32,
  float16: isnan_u16,
};

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  float16: 2,
};

export function wasmIsnan(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  const outRegion = wasmMalloc(size); // u8 output
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'bool',
    outRegion,
    size,
    Uint8Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
