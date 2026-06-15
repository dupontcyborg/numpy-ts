/**
 * WASM-accelerated element-wise logarithms (natural / base-2 / base-10).
 *
 * Unary: out[i] = log(a[i]) (or log2 / log10)
 * Float-only fast path: float64/float32 use native SIMD kernels, float16 routes
 * through the f32 kernel. Integer/complex inputs return null so the caller keeps
 * the existing JS fallback (NumPy promotes ints to float64, handled there).
 */

import { effectiveDType, isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import * as logBase from './bins/log.wasm';
import * as logRelaxed from './bins/log-relaxed.wasm';
import { wasmConfig } from './config';
import { useRelaxedKernels } from './detect';
import {
  f16InputToScratchF32,
  f32OutputToF16Region,
  resetScratchAllocator,
  resolveInputPtr,
  wasmMalloc,
} from './runtime';

const BASE_THRESHOLD = 32;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

// Pick baseline vs relaxed-SIMD (FMA) kernels once, lazily — the benchmark
// runner sets wasmConfig.useRelaxedSimd before the first op.
let _bins: typeof logBase | null = null;
function bins(): typeof logBase {
  _bins ??= useRelaxedKernels() ? logRelaxed : logBase;
  return _bins;
}

/**
 * Shared float-only WASM log driver. `f64`/`f32` are the matching base kernels.
 * Returns null when WASM can't handle the case (non-contiguous, too small,
 * complex, or a non-float dtype).
 *
 * `allowF64` gates the f64 path: V8's `Math.log` is so well optimized that the
 * 2-wide SIMD kernel + JS↔WASM boundary loses to it for natural log, so
 * `wasmLog` keeps f64 on the JS fallback. `Math.log2`/`Math.log10` are slower in
 * V8, so log2/log10 enable the f64 kernel and win.
 */
function wasmLogBase(
  a: ArrayStorage,
  f64: UnaryFn,
  f32: UnaryFn,
  allowF64: boolean,
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  if (isComplexDType(dtype)) return null;

  // float16: convert to f32 scratch, run f32 kernel, convert back to f16.
  if (dtype === 'float16') {
    const outRegion = wasmMalloc(size * 4);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = f16InputToScratchF32(a, size);
    f32(aPtr, outRegion.ptr, size);

    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      f16Region,
      size,
      Float16Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray,
    );
  }

  // Native float path (f32/f64). Integers fall through to the JS fallback.
  if (dtype !== 'float32' && dtype !== 'float64') return null;
  if (dtype === 'float64' && !allowF64) return null;
  const isF32 = dtype === 'float32';
  const bpe = isF32 ? 4 : 8;
  const Ctor = isF32 ? Float32Array : Float64Array;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  (isF32 ? f32 : f64)(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );
}

/** WASM natural log (f32/f16 only; f64 stays on the faster JS Math.log). */
export function wasmLog(a: ArrayStorage): ArrayStorage | null {
  return wasmLogBase(a, bins().log_f64, bins().log_f32, false);
}

/** WASM base-2 log. Returns null if WASM can't handle this case. */
export function wasmLog2(a: ArrayStorage): ArrayStorage | null {
  return wasmLogBase(a, bins().log2_f64, bins().log2_f32, true);
}

/** WASM base-10 log. Returns null if WASM can't handle this case. */
export function wasmLog10(a: ArrayStorage): ArrayStorage | null {
  return wasmLogBase(a, bins().log10_f64, bins().log10_f32, true);
}
