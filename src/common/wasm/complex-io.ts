/**
 * Shared I/O plumbing for complex unary WASM kernels.
 *
 * Each op wrapper (exp/exp2/sin/cos/log) owns its complex kernels (e.g.
 * `exp_c128`, `log2_c64`) in its own bin; this helper handles the common
 * interleaved-[re,im] region allocation, pointer resolution, and result
 * wrapping. The kernel processes 2 complex elements per SIMD step.
 */

import type { TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr, wasmMalloc } from './runtime';

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const BASE_THRESHOLD = 32;

/**
 * Run a complex unary kernel `${op}_c128` / `${op}_c64` from the given bin
 * namespace. Returns null when WASM can't handle the case (non-contiguous, too
 * small, or non-complex) so the caller keeps its JS fallback.
 */
export function complexUnaryWasm(
  a: ArrayStorage,
  op: string,
  bins: Record<string, UnaryFn>,
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype !== 'complex128' && dtype !== 'complex64') return null;

  const isC64 = dtype === 'complex64';
  const bpe = isC64 ? 4 : 8; // per real component
  const Ctor = isC64 ? Float32Array : Float64Array;
  const totalElements = size * 2; // re + im per complex

  const outRegion = wasmMalloc(totalElements * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, totalElements, bpe);

  bins[`${op}_${isC64 ? 'c64' : 'c128'}`]!(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );
}
