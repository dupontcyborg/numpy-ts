/**
 * WASM-accelerated element-wise deg2rad and rad2deg.
 *
 * deg2rad: out[i] = a[i] * (π / 180)
 * rad2deg: out[i] = a[i] * (180 / π)
 * Returns null if WASM can't handle this case.
 * Only supports float64 and float32 — the ops layer handles int→float promotion.
 */

import { deg2rad_f64, deg2rad_f32, rad2deg_f64, rad2deg_f32 } from './bins/deg2rad.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const deg2radKernels: Partial<Record<DType, UnaryFn>> = {
  float64: deg2rad_f64,
  float32: deg2rad_f32,
};

const rad2degKernels: Partial<Record<DType, UnaryFn>> = {
  float64: rad2deg_f64,
  float32: rad2deg_f32,
};

function runUnary(a: ArrayStorage, kernels: Partial<Record<DType, UnaryFn>>): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  if (!kernel) return null;

  const isF32 = dtype === 'float32';
  const bpe = isF32 ? 4 : 8;
  const Ctor = isF32 ? Float32Array : Float64Array;
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  kernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}

/**
 * WASM-accelerated element-wise deg2rad.
 * Returns null if WASM can't handle (non-float, non-contiguous, too small).
 */
export function wasmDeg2rad(a: ArrayStorage): ArrayStorage | null {
  return runUnary(a, deg2radKernels);
}

/**
 * WASM-accelerated element-wise rad2deg.
 * Returns null if WASM can't handle (non-float, non-contiguous, too small).
 */
export function wasmRad2deg(a: ArrayStorage): ArrayStorage | null {
  return runUnary(a, rad2degKernels);
}
