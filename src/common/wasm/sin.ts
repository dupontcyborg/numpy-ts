/**
 * WASM-accelerated element-wise sine.
 *
 * Unary: out[i] = sin(a[i])
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types are converted to float64
 * in JS and run through the f64 SIMD kernel (matches NumPy's promotion).
 */

import { sin_f64, sin_f32 } from './bins/sin.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchCopyIn,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, isBigIntDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: sin_f64,
  float32: sin_f32,
};

/**
 * WASM-accelerated element-wise sine.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmSin(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // float16 path: convert to f32, run f32 kernel, convert back
  if (dtype === 'float16') {
    const bpe = 4; // f32
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = f16InputToScratchF32(a, size);

    sin_f32(aPtr, outRegion.ptr, size);

    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  // Native float path
  const nativeKernel = kernels[dtype];
  if (nativeKernel) {
    const isF32 = dtype === 'float32';
    const bpe = isF32 ? 4 : 8;
    const Ctor = isF32 ? Float32Array : Float64Array;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

    nativeKernel(aPtr, outRegion.ptr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
  }

  // Integer path: convert to float64, run f64 kernel
  const bpe = 8; // f64
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aOff = a.offset;
  const src = a.data;
  const converted = new Float64Array(size);
  if (isBigIntDType(dtype)) {
    for (let i = 0; i < size; i++) converted[i] = Number(src[aOff + i]!);
  } else {
    for (let i = 0; i < size; i++) converted[i] = src[aOff + i] as number;
  }

  const aPtr = scratchCopyIn(converted as unknown as TypedArray);

  sin_f64(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
