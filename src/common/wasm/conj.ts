/**
 * WASM-accelerated element-wise complex conjugate.
 *
 * Unary: out[re] = a[re], out[im] = -a[im]
 * Returns null if WASM can't handle this case.
 */

import { conj_c64, conj_c128 } from './bins/conj.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  complex64: conj_c64,
  complex128: conj_c128,
};

/**
 * WASM-accelerated element-wise complex conjugate.
 * Returns null if WASM can't handle.
 */
export function wasmConj(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  const kernel = kernels[dtype];
  if (!kernel) return null;

  const bpe = dtype === 'complex64' ? 4 : 8; // f32 vs f64
  const Ctor = dtype === 'complex64' ? Float32Array : Float64Array;
  const totalElements = size * 2;
  const outBytes = totalElements * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, totalElements, bpe);

  kernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => InstanceType<typeof Float32Array | typeof Float64Array>
  );
}
