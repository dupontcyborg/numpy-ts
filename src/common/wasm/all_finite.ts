/**
 * WASM-accelerated "all finite" check.
 * Returns true if every element is finite, false if any NaN or Inf found.
 * Early-exits on first non-finite value.
 */

import { type DType, effectiveDType } from '../dtype';
import { ArrayStorage } from '../storage';
import { all_finite_f32, all_finite_f64, all_finite_u16 } from './bins/all_finite.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr } from './runtime';

type AllFiniteFn = (aPtr: number, N: number) => number;

const kernels: Partial<Record<DType, AllFiniteFn>> = {
  float64: all_finite_f64,
  float32: all_finite_f32,
  float16: all_finite_u16,
};

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  float16: 2,
};

export function wasmAllFinite(a: ArrayStorage): boolean | null {
  if (!a.isCContiguous) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  const size = a.size;
  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  return kernel(aPtr, size) === 1;
}
