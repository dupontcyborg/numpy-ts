/**
 * WASM-accelerated bitwise_count (population count).
 *
 * Unary: out[i] = popcount(a[i])
 * Output is always uint8.
 * For signed types, counts bits of abs(value) to match NumPy.
 * Returns null if WASM can't handle this case.
 */

import {
  bitwise_count_i64,
  bitwise_count_u64,
  bitwise_count_i32,
  bitwise_count_u32,
  bitwise_count_i16,
  bitwise_count_u16,
  bitwise_count_i8,
  bitwise_count_u8,
} from './bins/bitwise_count.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type CountFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, CountFn>> = {
  int64: bitwise_count_i64,
  uint64: bitwise_count_u64,
  int32: bitwise_count_i32,
  uint32: bitwise_count_u32,
  int16: bitwise_count_i16,
  uint16: bitwise_count_u16,
  int8: bitwise_count_i8,
  uint8: bitwise_count_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

export function wasmBitwiseCount(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size; // output is uint8, 1 byte per element

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size);

  const outData = copyOut(outPtr, size, Uint8Array as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray);

  return ArrayStorage.fromData(outData, Array.from(a.shape), 'uint8');
}
