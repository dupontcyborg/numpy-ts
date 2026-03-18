/**
 * WASM-accelerated element-wise array reversal.
 *
 * Unary: out[i] = a[N-1-i]
 * Returns null if WASM can't handle this case.
 */

import { flip_f64, flip_f32, flip_i64, flip_i32, flip_i16, flip_i8 } from './bins/flip.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: flip_f64,
  float32: flip_f32,
  int64: flip_i64,
  uint64: flip_i64,
  int32: flip_i32,
  uint32: flip_i32,
  int16: flip_i16,
  uint16: flip_i16,
  int8: flip_i8,
  uint8: flip_i8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
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
 * WASM-accelerated full array reversal.
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmFlip(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
