/**
 * WASM-accelerated element-wise square.
 *
 * Unary: out[i] = a[i] * a[i]
 * Returns null if WASM can't handle this case.
 */

import {
  square_f64,
  square_f32,
  square_i64,
  square_i32,
  square_i16,
  square_i8,
  square_c128,
  square_c64,
} from './bins/square.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: square_f64,
  float32: square_f32,
  // float16 excluded: f16→f32 conversion overhead makes JS path faster
  complex128: square_c128,
  complex64: square_c64,
  int64: square_i64,
  uint64: square_i64,
  int32: square_i32,
  uint32: square_i32,
  int16: square_i16,
  uint16: square_i16,
  int8: square_i8,
  uint8: square_i8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  complex128: Float64Array,
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

/**
 * WASM-accelerated element-wise square.
 * Returns null if WASM can't handle (non-contiguous, too small).
 */
export function wasmSquare(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  // Complex: interleaved re/im pairs, so data length = size * 2
  const complexFactor = isComplexDType(dtype) ? 2 : 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = size * complexFactor;
  const outBytes = totalElements * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * complexFactor,
    totalElements,
    bpe
  );

  kernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
