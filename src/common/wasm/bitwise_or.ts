/**
 * WASM-accelerated element-wise bitwise OR.
 *
 * Binary: out[i] = a[i] | b[i]  (same-shape contiguous arrays)
 * Returns null if WASM can't handle this case.
 */

import {
  bitwise_or_i64,
  bitwise_or_i32,
  bitwise_or_i16,
  bitwise_or_i8,
} from './bins/bitwise_or.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, BinaryFn>> = {
  int64: bitwise_or_i64,
  uint64: bitwise_or_i64,
  int32: bitwise_or_i32,
  uint32: bitwise_or_i32,
  int16: bitwise_or_i16,
  uint16: bitwise_or_i16,
  int8: bitwise_or_i8,
  uint8: bitwise_or_i8,
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

export function wasmBitwiseOr(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const bBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const bOff = b.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
  const bData = b.data.subarray(bOff, bOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
