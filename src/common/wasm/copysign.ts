/**
 * WASM-accelerated element-wise copysign.
 *
 * Binary: out[i] = copysign(x1[i], x2[i])  (magnitude of x1, sign of x2)
 * Scalar: out[i] = copysign(x1[i], scalar)
 * Output is always float64. Returns null if WASM can't handle.
 */

import {
  copysign_f64,
  copysign_f32,
  copysign_i64,
  copysign_i32,
  copysign_i16,
  copysign_i8,
  copysign_scalar_f64,
  copysign_scalar_f32,
  copysign_scalar_i64,
  copysign_scalar_i32,
  copysign_scalar_i16,
  copysign_scalar_i8,
} from './bins/copysign.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (x1Ptr: number, x2Ptr: number, outPtr: number, N: number) => void;
type ScalarFn = (x1Ptr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: copysign_f64,
  float32: copysign_f32,
  int64: copysign_i64,
  int32: copysign_i32,
  int16: copysign_i16,
  int8: copysign_i8,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: copysign_scalar_f64,
  float32: copysign_scalar_f32,
  int64: copysign_scalar_i64,
  int32: copysign_scalar_i32,
  int16: copysign_scalar_i16,
  int8: copysign_scalar_i8,
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
 * WASM-accelerated copysign of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmCopysign(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage | null {
  if (!x1.isCContiguous || !x2.isCContiguous) return null;

  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = x1.dtype;
  const kernel = binaryKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;
  if (x2.dtype !== dtype) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const inBytes = size * bpe;
  const outBytes = size * 8; // f64 output

  ensureMemory(inBytes * 2 + outBytes);
  resetAllocator();

  const x1Ptr = copyIn(x1.data.subarray(x1.offset, x1.offset + size) as TypedArray);
  const x2Ptr = copyIn(x2.data.subarray(x2.offset, x2.offset + size) as TypedArray);
  const outPtr = alloc(outBytes);

  kernel(x1Ptr, x2Ptr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(x1.shape), 'float64');
}

/**
 * WASM-accelerated copysign of array and scalar.
 * Returns null if WASM can't handle.
 */
export function wasmCopysignScalar(x1: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!x1.isCContiguous) return null;

  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = x1.dtype;
  const kernel = scalarKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const inBytes = size * bpe;
  const outBytes = size * 8; // f64 output

  ensureMemory(inBytes + outBytes);
  resetAllocator();

  const x1Ptr = copyIn(x1.data.subarray(x1.offset, x1.offset + size) as TypedArray);
  const outPtr = alloc(outBytes);

  kernel(x1Ptr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(x1.shape), 'float64');
}
