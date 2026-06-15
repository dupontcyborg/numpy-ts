/**
 * WASM-accelerated element-wise logarithms (natural / base-2 / base-10).
 *
 * Unary: out[i] = log(a[i]) (or log2 / log10)
 * Float fast path: float64/float32 use native SIMD kernels, float16 routes
 * through the f32 kernel. Integers use type-appropriate output (NumPy promotion):
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 * Complex inputs return null so the caller keeps the JS fallback.
 */

import { type DType, effectiveDType, hasFloat16, isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import * as logBase from './bins/log.wasm';
import * as logRelaxed from './bins/log-relaxed.wasm';
import { complexUnaryWasm } from './complex-io';
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
function bins(): Record<string, UnaryFn> {
  _bins ??= useRelaxedKernels() ? logRelaxed : logBase;
  return _bins as unknown as Record<string, UnaryFn>;
}

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

// int dtype → kernel suffix
const f64IntSuffix: Partial<Record<DType, string>> = {
  int64: 'i64_f64',
  uint64: 'u64_f64',
  int32: 'i32_f64',
  uint32: 'u32_f64',
};
const f32IntSuffix: Partial<Record<DType, string>> = {
  int16: 'i16_f32',
  uint16: 'u16_f32',
  int8: 'i8_f32',
  uint8: 'u8_f32',
};

/**
 * Shared WASM log driver for one base. `op` is the kernel prefix ('log' /
 * 'log2' / 'log10'); kernels are looked up by `${op}_${suffix}`.
 *
 * `allowF64` gates ONLY the float64-input path: V8's `Math.log` beats a 2-wide
 * SIMD kernel + JS↔WASM boundary for natural log, so `wasmLog` keeps float64
 * INPUT on the JS fallback. Integer inputs always use WASM (NumPy promotes them
 * to float anyway, and the JS path pays a per-element Number() conversion —
 * expensive for int64/uint64 BigInts).
 */
function wasmLogImpl(a: ArrayStorage, op: string, allowF64: boolean): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  if (isComplexDType(dtype)) return complexUnaryWasm(a, op, bins());

  const k = bins();

  // float16: convert to f32 scratch, run f32 kernel, convert back to f16.
  if (dtype === 'float16') {
    const outRegion = wasmMalloc(size * 4);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = f16InputToScratchF32(a, size);
    k[`${op}_f32`]!(aPtr, outRegion.ptr, size);

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

  // Native float path (f32/f64). float64 input is gated by allowF64.
  if (dtype === 'float32' || dtype === 'float64') {
    if (dtype === 'float64' && !allowF64) return null;
    const isF32 = dtype === 'float32';
    const bpe = isF32 ? 4 : 8;
    const Ctor = isF32 ? Float32Array : Float64Array;

    const outRegion = wasmMalloc(size * bpe);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    k[`${op}_${isF32 ? 'f32' : 'f64'}`]!(aPtr, outRegion.ptr, size);

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

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16.
  const smallSuffix = f32IntSuffix[dtype];
  if (smallSuffix) {
    const outRegion = wasmMalloc(size * 4); // f32
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);
    k[`${op}_${smallSuffix}`]!(aPtr, outRegion.ptr, size);

    // i8/u8 → downcast f32 to f16 (matching NumPy's int8/uint8 → float16)
    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
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

  // Large int path: i32/u32/i64/u64 → f64 output.
  const largeSuffix = f64IntSuffix[dtype];
  if (!largeSuffix) return null;

  const outRegion = wasmMalloc(size * 8);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);
  k[`${op}_${largeSuffix}`]!(aPtr, outRegion.ptr, size);

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

/** WASM natural log (f32/f16/int only; f64 input stays on the faster JS Math.log). */
export function wasmLog(a: ArrayStorage): ArrayStorage | null {
  return wasmLogImpl(a, 'log', false);
}

/** WASM base-2 log. Returns null if WASM can't handle this case. */
export function wasmLog2(a: ArrayStorage): ArrayStorage | null {
  return wasmLogImpl(a, 'log2', true);
}

/** WASM base-10 log. Returns null if WASM can't handle this case. */
export function wasmLog10(a: ArrayStorage): ArrayStorage | null {
  return wasmLogImpl(a, 'log10', true);
}
