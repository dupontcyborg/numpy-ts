/**
 * WASM-accelerated element-wise hyperbolic tangent.
 *
 * Unary: out[i] = tanh(a[i])
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types are converted to float64
 * in JS and run through the f64 SIMD kernel (matches NumPy's promotion).
 */

import {
  tanh_f64,
  tanh_f32,
  tanh_i64,
  tanh_u64,
  tanh_i32_f64,
  tanh_u32_f64,
  tanh_i16_f64,
  tanh_u16_f64,
  tanh_i8_f64,
  tanh_u8_f64,
} from './bins/tanh.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: tanh_f64,
  float32: tanh_f32,
};

const intKernels: Partial<Record<DType, UnaryFn>> = {
  int64: tanh_i64,
  uint64: tanh_u64,
  int32: tanh_i32_f64,
  uint32: tanh_u32_f64,
  int16: tanh_i16_f64,
  uint16: tanh_u16_f64,
  int8: tanh_i8_f64,
  uint8: tanh_u8_f64,
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
 * WASM-accelerated element-wise hyperbolic tangent.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmTanh(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // float16 path: convert to f32, run f32 kernel, convert back
  if (dtype === 'float16') {
    const bpe = 4;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = f16InputToScratchF32(a, size);

    tanh_f32(aPtr, outRegion.ptr, size);

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
