/**
 * WASM-accelerated element-wise frexp (decompose into mantissa + exponent).
 *
 * Returns [mantissa, exponent] where x = mantissa * 2^exponent.
 * Mantissa preserves input dtype (f64→f64, f32→f32, f16→f16).
 * Returns null if WASM can't handle this case.
 */

import { frexp_f64, frexp_f32 } from './bins/frexp.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, hasFloat16, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

export function wasmFrexp(a: ArrayStorage): [ArrayStorage, ArrayStorage] | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  if (dtype !== 'float64' && dtype !== 'float32' && dtype !== 'float16') return null;

  const bpe_i32 = 4;

  // Allocate exponent region (always int32)
  const eRegion = wasmMalloc(size * bpe_i32);
  if (!eRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (dtype === 'float16') {
    // f16 → run f32 kernel → downcast mantissa to f16
    const mRegion = wasmMalloc(size * 4);
    if (!mRegion) {
      eRegion.release();
      return null;
    }

    const aPtr = f16InputToScratchF32(a, size);
    frexp_f32(aPtr, mRegion.ptr, eRegion.ptr, size);

    if (hasFloat16) {
      const f16Region = f32OutputToF16Region(mRegion, size);
      mRegion.release();
      if (!f16Region) {
        eRegion.release();
        return null;
      }
      const mantissa = ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      );
      const exponent = ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'int32',
        eRegion,
        size,
        Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      );
      return [mantissa, exponent];
    }
    // No Float16Array — return as f32
    const mantissa = ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      mRegion,
      size,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    const exponent = ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'int32',
      eRegion,
      size,
      Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return [mantissa, exponent];
  }

  if (dtype === 'float32') {
    const mRegion = wasmMalloc(size * 4);
    if (!mRegion) {
      eRegion.release();
      return null;
    }

    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 4);
    frexp_f32(aPtr, mRegion.ptr, eRegion.ptr, size);

    const mantissa = ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      mRegion,
      size,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    const exponent = ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'int32',
      eRegion,
      size,
      Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return [mantissa, exponent];
  }

  // float64
  const mRegion = wasmMalloc(size * 8);
  if (!mRegion) {
    eRegion.release();
    return null;
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 8);
  frexp_f64(aPtr, mRegion.ptr, eRegion.ptr, size);

  const mantissa = ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    mRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  const exponent = ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'int32',
    eRegion,
    size,
    Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return [mantissa, exponent];
}
