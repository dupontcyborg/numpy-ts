/**
 * WASM-accelerated element-wise frexp (decompose into mantissa + exponent).
 *
 * Returns [mantissa, exponent] where x = mantissa * 2^exponent.
 * Returns null if WASM can't handle this case.
 */

import { frexp_f64 } from './bins/frexp.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr, scratchCopyIn } from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

export function wasmFrexp(a: ArrayStorage): [ArrayStorage, ArrayStorage] | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // frexp only defined for float types; integers are promoted to float64 by caller
  const dtype = effectiveDType(a.dtype);
  if (dtype !== 'float64' && dtype !== 'float32' && dtype !== 'float16') return null;

  const bpe_f64 = 8;
  const bpe_i32 = 4;

  // Allocate persistent regions for both outputs
  const mRegion = wasmMalloc(size * bpe_f64);
  if (!mRegion) return null;
  const eRegion = wasmMalloc(size * bpe_i32);
  if (!eRegion) {
    mRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();

  // If float16/float32, we need to promote to f64 for the kernel
  let aPtr: number;
  if (dtype === 'float16') {
    const f32Data = new Float32Array(
      a.data.subarray(a.offset, a.offset + size) as unknown as ArrayLike<number>
    );
    const f64Data = new Float64Array(size);
    for (let i = 0; i < size; i++) f64Data[i] = f32Data[i]!;
    aPtr = scratchCopyIn(f64Data as unknown as TypedArray);
  } else if (dtype === 'float32') {
    const f32Data = a.data.subarray(a.offset, a.offset + size) as Float32Array;
    const f64Data = new Float64Array(size);
    for (let i = 0; i < size; i++) f64Data[i] = f32Data[i]!;
    aPtr = scratchCopyIn(f64Data as unknown as TypedArray);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe_f64);
  }

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
