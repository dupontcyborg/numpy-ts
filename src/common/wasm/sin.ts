/**
 * WASM-accelerated element-wise sine.
 *
 * Unary: out[i] = sin(a[i])
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types are converted to float64
 * in JS and run through the f64 SIMD kernel (matches NumPy's promotion).
 */

import {
  sin_f64,
  sin_f32,
  sin_i64_f64,
  sin_u64_f64,
  sin_i32_f64,
  sin_u32_f64,
  sin_i16_f64,
  sin_u16_f64,
  sin_i8_f64,
  sin_u8_f64,
} from './bins/sin.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: sin_f64,
  float32: sin_f32,
};

const intKernels: Partial<Record<DType, UnaryFn>> = {
  int64: sin_i64_f64,
  uint64: sin_u64_f64,
  int32: sin_i32_f64,
  uint32: sin_u32_f64,
  int16: sin_i16_f64,
  uint16: sin_u16_f64,
  int8: sin_i8_f64,
  uint8: sin_u8_f64,
};

const bpeMap: Partial<Record<DType, number>> = {
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
};

/**
 * WASM-accelerated element-wise sine.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmSin(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
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

  // Integer path: Zig kernel reads native int type, converts to f64 internally
  const intKernel = intKernels[dtype];
  const inBpe = bpeMap[dtype];
  if (!intKernel || !inBpe) return null;

  const outBytes = size * 8;
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);

  intKernel(aPtr, outRegion.ptr, size);

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
