/**
 * WASM-accelerated element-wise logical OR.
 *
 * Binary: out[i] = (a[i] != 0) | (b[i] != 0)
 * Scalar: out[i] = (a[i] != 0) | (scalar != 0)
 * Output is always bool (Uint8Array). Returns null if WASM can't handle.
 */

import {
  logical_or_f64,
  logical_or_f32,
  logical_or_i64,
  logical_or_i32,
  logical_or_i16,
  logical_or_i8,
  logical_or_scalar_f64,
  logical_or_scalar_f32,
  logical_or_scalar_i64,
  logical_or_scalar_i32,
  logical_or_scalar_i16,
  logical_or_scalar_i8,
} from './bins/logical_or.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: logical_or_f64,
  float32: logical_or_f32,
  int64: logical_or_i64,
  uint64: logical_or_i64,
  int32: logical_or_i32,
  uint32: logical_or_i32,
  int16: logical_or_i16,
  uint16: logical_or_i16,
  int8: logical_or_i8,
  uint8: logical_or_i8 as unknown as BinaryFn,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: logical_or_scalar_f64,
  float32: logical_or_scalar_f32,
  int64: logical_or_scalar_i64,
  uint64: logical_or_scalar_i64,
  int32: logical_or_scalar_i32,
  uint32: logical_or_scalar_i32,
  int16: logical_or_scalar_i16,
  uint16: logical_or_scalar_i16,
  int8: logical_or_scalar_i8,
  uint8: logical_or_scalar_i8 as unknown as ScalarFn,
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
 * WASM-accelerated logical OR of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmLogicalOr(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = binaryKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;
  if (b.dtype !== dtype) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const bBytes = size * bpe;
  const outBytes = size;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const bPtr = copyIn(b.data.subarray(b.offset, b.offset + size) as TypedArray);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Uint8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), 'bool');
}

/**
 * WASM-accelerated logical OR of array and scalar.
 * Returns null if WASM can't handle.
 */
export function wasmLogicalOrScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = scalarKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    size,
    Uint8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), 'bool');
}
