/**
 * WASM-accelerated element-wise exponential.
 *
 * Unary: out[i] = exp(a[i])
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types use type-appropriate output:
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 */

import { type DType, effectiveDType, hasFloat16, isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import * as expBase from './bins/exp.wasm';
import * as expRelaxed from './bins/exp-relaxed.wasm';
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
let _bins: typeof expBase | null = null;
function bins(): typeof expBase {
  _bins ??= useRelaxedKernels() ? expRelaxed : expBase;
  return _bins;
}

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: (...a) => bins().exp_f64(...a),
  float32: (...a) => bins().exp_f32(...a),
};

// Large int → f64 output (i32/u32/i64/u64 need f64 precision)
const largeIntKernels: Partial<Record<DType, UnaryFn>> = {
  int64: (...a) => bins().exp_i64_f64(...a),
  uint64: (...a) => bins().exp_u64_f64(...a),
  int32: (...a) => bins().exp_i32_f64(...a),
  uint32: (...a) => bins().exp_u32_f64(...a),
};

// Small int → f32 output (i8/u8/i16/u16 → f32, then optionally downcast to f16)
const smallIntKernels: Partial<Record<DType, UnaryFn>> = {
  int16: (...a) => bins().exp_i16_f32(...a),
  uint16: (...a) => bins().exp_u16_f32(...a),
  int8: (...a) => bins().exp_i8_f32(...a),
  uint8: (...a) => bins().exp_u8_f32(...a),
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
 * WASM-accelerated element-wise exponential.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmExp(a: ArrayStorage): ArrayStorage | null {
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

    bins().exp_f32(aPtr, outRegion.ptr, size);

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

  // Native float path (f32/f64)
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
      Ctor as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  const inBpe = bpeMap[dtype];
  if (!inBpe) return null;

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntKernels[dtype];
  if (smallKernel) {
    const outBytes = size * 4; // f32
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);

    smallKernel(aPtr, outRegion.ptr, size);

    // i8/u8 → downcast f32 to f16 (matching NumPy's bool/int8/uint8 → float16)
    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8' || dtype === 'bool')) {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (
          buf: ArrayBuffer,
          off: number,
          len: number,
        ) => TypedArray,
      );
    }

    // i16/u16 → f32 output
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      outRegion,
      size,
      Float32Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  // Large int path: i32/u32/i64/u64 → f64 output
  const largeKernel = largeIntKernels[dtype];
  if (!largeKernel) return null;

  const outBytes = size * 8;
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);

  largeKernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );
}
