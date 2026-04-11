/**
 * WASM-accelerated element-wise logical AND.
 *
 * Binary: out[i] = (a[i] != 0) & (b[i] != 0)
 * Scalar: out[i] = (a[i] != 0) & (scalar != 0)
 * Output is always bool (Uint8Array). Returns null if WASM can't handle.
 */

import {
  logical_and_f64,
  logical_and_f32,
  logical_and_i64,
  logical_and_i32,
  logical_and_i16,
  logical_and_i8,
  logical_and_scalar_f64,
  logical_and_scalar_f32,
  logical_and_scalar_i64,
  logical_and_scalar_i32,
  logical_and_scalar_i16,
  logical_and_scalar_i8,
  logical_and_f16,
  logical_and_scalar_f16,
} from './bins/logical_and.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { hasFloat16, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: logical_and_f64,
  float32: logical_and_f32,
  float16: logical_and_f16 as unknown as BinaryFn, // raw u16 bit check
  int64: logical_and_i64,
  uint64: logical_and_i64,
  int32: logical_and_i32,
  uint32: logical_and_i32,
  int16: logical_and_i16,
  uint16: logical_and_i16,
  int8: logical_and_i8,
  uint8: logical_and_i8 as unknown as BinaryFn,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: logical_and_scalar_f64,
  float32: logical_and_scalar_f32,
  float16: logical_and_scalar_f16 as unknown as ScalarFn, // raw u16 bit check
  int64: logical_and_scalar_i64,
  uint64: logical_and_scalar_i64,
  int32: logical_and_scalar_i32,
  uint32: logical_and_scalar_i32,
  int16: logical_and_scalar_i16,
  uint16: logical_and_scalar_i16,
  int8: logical_and_scalar_i8,
  uint8: logical_and_scalar_i8 as unknown as ScalarFn,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const inputCtorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: (typeof Float16Array !== 'undefined'
    ? Float16Array
    : Float32Array) as unknown as AnyTypedArrayCtor,
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
 * WASM-accelerated logical AND of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmLogicalAnd(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype === 'float16' && !hasFloat16) return null;
  // Both arrays must have same dtype for WASM fast path
  if (b.dtype !== dtype) return null;

  const kernel = binaryKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size; // u8 output

  // Allocate output in persistent WASM heap
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  // Resolve input pointers (zero-copy if WASM-backed, scratch-copy if JS)
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);

  kernel(aPtr, bPtr, outRegion.ptr, size);

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

/**
 * WASM-accelerated logical AND of array and scalar.
 * Returns null if WASM can't handle.
 */
export function wasmLogicalAndScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype === 'float16' && !hasFloat16) return null;

  const kernel = scalarKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  if (!kernel || !InCtor) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size; // u8 output

  // Allocate output in persistent WASM heap
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  // Resolve input pointer
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  kernel(aPtr, outRegion.ptr, size, scalar);

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
