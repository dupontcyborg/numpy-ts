/**
 * WASM-accelerated element-wise addition.
 *
 * Binary: out[i] = a[i] + b[i]  (same-shape contiguous arrays)
 * Scalar: out[i] = a[i] + scalar
 * Returns null if WASM can't handle this case.
 */

import {
  add_f64,
  add_f32,
  add_i64,
  add_i32,
  add_i16,
  add_i8,
  add_c128,
  add_c64,
  add_scalar_f64,
  add_scalar_f32,
  add_scalar_i64,
  add_scalar_i32,
  add_scalar_i16,
  add_scalar_i8,
  add_scalar_c128,
  add_scalar_c64,
} from './bins/add.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: add_f64,
  float32: add_f32,
  // float16 excluded: f16→f32 conversion overhead makes JS path faster
  int64: add_i64,
  uint64: add_i64,
  int32: add_i32,
  uint32: add_i32,
  int16: add_i16,
  uint16: add_i16,
  int8: add_i8,
  uint8: add_i8,
  complex128: add_c128,
  complex64: add_c64,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: add_scalar_f64,
  float32: add_scalar_f32,
  // float16 excluded: f16→f32 conversion overhead makes JS path faster
  int64: add_scalar_i64,
  uint64: add_scalar_i64,
  int32: add_scalar_i32,
  uint32: add_scalar_i32,
  int16: add_scalar_i16,
  uint16: add_scalar_i16,
  int8: add_scalar_i8,
  uint8: add_scalar_i8,
  complex128: add_scalar_c128,
  complex64: add_scalar_c64,
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

const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated element-wise add of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmAdd(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);

  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[dtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = size * factor;
  const outBytes = totalElements * bpe;

  // Allocate output in persistent WASM heap
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  // Resolve input pointers (zero-copy if WASM-backed, scratch-copy if JS)
  resetScratchAllocator();
  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * factor,
    totalElements,
    bpe
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * factor,
    totalElements,
    bpe
  );

  kernel(aPtr, bPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}

/**
 * WASM-accelerated element-wise add of array + scalar.
 * Returns null if WASM can't handle.
 */
export function wasmAddScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[dtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = size * factor;
  const outBytes = totalElements * bpe;

  // Allocate output in persistent WASM heap
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  // Resolve input pointer
  resetScratchAllocator();
  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * factor,
    totalElements,
    bpe
  );

  kernel(aPtr, outRegion.ptr, size, scalar);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
