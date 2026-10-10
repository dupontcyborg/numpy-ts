/**
 * WASM-accelerated L2 vector norm: sqrt(sum(x[i]^2)).
 * Returns null if WASM can't handle.
 */

import { type DType, effectiveDType, isComplexDType } from '../dtype';
import type { ArrayStorage } from '../storage';
import * as floatBase from './bins/vector_norm.wasm';
import { wasmConfig } from './config';
import { f16InputToScratchF32, resetScratchAllocator, resolveInputPtr } from './runtime';

function float(): typeof floatBase {
  return floatBase;
}

const BASE_THRESHOLD = 32;

type NormFn = (aPtr: number, N: number) => number;

const kernels: Partial<Record<DType, { fn: NormFn; bpe: number }>> = {
  float64: { fn: (...a) => float().vector_norm2_f64(...a), bpe: 8 },
  float32: { fn: (...a) => float().vector_norm2_f32(...a), bpe: 4 },
};

/**
 * WASM-accelerated L2 norm (Euclidean norm).
 *
 * Complex input runs on the real kernels over twice as many slots, which is not
 * an approximation: sum of |z|^2 is sum of (re^2 + im^2), and that is exactly
 * the squared norm of the interleaved [re, im] buffer read as reals. A separate
 * complex kernel would compute the same sum from the same bytes.
 *
 * @param a - Input array
 * @returns sqrt(sum(|x|^2)), or null if WASM can't handle it
 */
export function wasmVectorNorm2(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  if (isComplexDType(a.dtype)) {
    const isC64 = a.dtype === 'complex64';
    const bpe = isC64 ? 4 : 8;
    const slots = size * 2;
    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const ptr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, slots, bpe);
    return isC64 ? float().vector_norm2_f32(ptr, slots) : float().vector_norm2_f64(ptr, slots);
  }

  const dtype = effectiveDType(a.dtype);

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Float16: convert to f32 and use f32 kernel
  if (dtype === 'float16') {
    const aPtr = f16InputToScratchF32(a, size);
    return float().vector_norm2_f32(aPtr, size);
  }

  const entry = kernels[dtype];
  if (!entry) {
    wasmConfig.wasmCallCount--; // undo increment
    return null;
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, entry.bpe);
  return entry.fn(aPtr, size);
}
