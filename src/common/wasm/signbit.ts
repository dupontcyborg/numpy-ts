/**
 * WASM-accelerated element-wise signbit with SIMD.
 *
 * Unary: out[i] = 1 if a[i] has negative sign bit, 0 otherwise
 * Returns a boolean (uint8) array.
 * For unsigned types, signbit is always 0 — returns zeros directly.
 */

import {
  signbit_f64,
  signbit_f32,
  signbit_f16,
  signbit_i64,
  signbit_i32,
  signbit_i16,
  signbit_i8,
} from './bins/signbit.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type SignbitFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, { fn: SignbitFn; bpe: number }>> = {
  float64: { fn: signbit_f64, bpe: 8 },
  float32: { fn: signbit_f32, bpe: 4 },
  float16: { fn: signbit_f16, bpe: 2 },
  int64: { fn: signbit_i64, bpe: 8 },
  int32: { fn: signbit_i32, bpe: 4 },
  int16: { fn: signbit_i16, bpe: 2 },
  int8: { fn: signbit_i8, bpe: 1 },
};

/**
 * WASM-accelerated element-wise signbit.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmSignbit(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // Unsigned types: signbit is always 0
  if (dtype === 'uint8' || dtype === 'uint16' || dtype === 'uint32' || dtype === 'uint64') {
    return ArrayStorage.zeros(Array.from(a.shape), 'bool');
  }

  const kernel = kernels[dtype];
  if (!kernel) return null;

  const outRegion = wasmMalloc(size); // 1 byte per element (uint8)
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, kernel.bpe);
  kernel.fn(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'bool',
    outRegion,
    size,
    Uint8Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
