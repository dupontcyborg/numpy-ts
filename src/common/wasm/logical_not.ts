/**
 * WASM-accelerated element-wise logical NOT.
 *
 * Unary: out[i] = (a[i] == 0) ? 1 : 0
 * Output is always bool (Uint8Array). Returns null if WASM can't handle.
 */

import {
  logical_not_f64,
  logical_not_f32,
  logical_not_i64,
  logical_not_i32,
  logical_not_i16,
  logical_not_i8,
} from './bins/logical_not.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: logical_not_f64,
  float32: logical_not_f32,
  int64: logical_not_i64,
  uint64: logical_not_i64,
  int32: logical_not_i32,
  uint32: logical_not_i32,
  int16: logical_not_i16,
  uint16: logical_not_i16,
  int8: logical_not_i8,
  uint8: logical_not_i8 as unknown as UnaryFn,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const inputCtorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
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
 * WASM-accelerated logical NOT.
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmLogicalNot(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size; // u8 output

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Uint8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), 'bool');
}
