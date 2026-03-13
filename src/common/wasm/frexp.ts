/**
 * WASM-accelerated element-wise frexp (decompose into mantissa + exponent).
 *
 * Returns [mantissa, exponent] where x = mantissa * 2^exponent.
 * Returns null if WASM can't handle this case.
 */

import { frexp_f64 } from './bins/frexp.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

export function wasmFrexp(a: ArrayStorage): [ArrayStorage, ArrayStorage] | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // frexp only defined for float types; integers are promoted to float64 by caller
  const dtype = a.dtype;
  if (dtype !== 'float64' && dtype !== 'float32') return null;

  // Input gets promoted to f64 for the kernel, output is always f64 mantissa + i32 exponent
  const bpe_f64 = 8;
  const bpe_i32 = 4;
  ensureMemory(size * bpe_f64 + size * bpe_f64 + size * bpe_i32);
  resetAllocator();

  // If float32, we need to promote to f64 for the kernel
  let aData: TypedArray;
  if (dtype === 'float32') {
    const f32Data = a.data.subarray(a.offset, a.offset + size) as Float32Array;
    const f64Data = new Float64Array(size);
    for (let i = 0; i < size; i++) f64Data[i] = f32Data[i]!;
    aData = f64Data;
  } else {
    aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
  }

  const aPtr = copyIn(aData);
  const mPtr = alloc(size * bpe_f64);
  const ePtr = alloc(size * bpe_i32);
  frexp_f64(aPtr, mPtr, ePtr, size);

  const mantissaData = copyOut(
    mPtr,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  const exponentData = copyOut(
    ePtr,
    size,
    Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  const mantissa = ArrayStorage.fromData(mantissaData, Array.from(a.shape), 'float64');
  const exponent = ArrayStorage.fromData(exponentData, Array.from(a.shape), 'int32');
  return [mantissa, exponent];
}
