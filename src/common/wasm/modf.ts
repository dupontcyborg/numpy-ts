/**
 * WASM-accelerated element-wise modf (split into fractional + integral parts).
 *
 * Returns [fractional, integral] where integral = trunc(x), fractional = x - trunc(x).
 * Both outputs preserve input dtype (f64→f64, f32→f32).
 * Returns null if WASM can't handle this case.
 */

import { modf_f64, modf_f32 } from './bins/modf.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

export function wasmModf(a: ArrayStorage): [ArrayStorage, ArrayStorage] | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype !== 'float64' && dtype !== 'float32') return null;

  if (dtype === 'float32') {
    const bpe = 4;
    const fracRegion = wasmMalloc(size * bpe);
    if (!fracRegion) return null;
    const intRegion = wasmMalloc(size * bpe);
    if (!intRegion) {
      fracRegion.release();
      return null;
    }

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    modf_f32(aPtr, fracRegion.ptr, intRegion.ptr, size);

    const fractional = ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      fracRegion,
      size,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    const integral = ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      intRegion,
      size,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return [fractional, integral];
  }

  // float64
  const bpe = 8;
  const fracRegion = wasmMalloc(size * bpe);
  if (!fracRegion) return null;
  const intRegion = wasmMalloc(size * bpe);
  if (!intRegion) {
    fracRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  modf_f64(aPtr, fracRegion.ptr, intRegion.ptr, size);

  const fractional = ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    fracRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  const integral = ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    intRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return [fractional, integral];
}
