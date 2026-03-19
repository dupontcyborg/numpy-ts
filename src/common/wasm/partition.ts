/**
 * WASM-accelerated in-place partition (quickselect).
 *
 * Partitions a contiguous 1D buffer so that element at kth position
 * is in its final sorted position.
 * Returns null if WASM can't handle this case.
 */

import {
  partition_f64,
  partition_f32,
  partition_i64,
  partition_u64,
  partition_i32,
  partition_u32,
  partition_i16,
  partition_u16,
  partition_i8,
  partition_u8,
  partition_slices_f64,
  partition_slices_f32,
  partition_slices_i64,
  partition_slices_u64,
  partition_slices_i32,
  partition_slices_u32,
  partition_slices_i16,
  partition_slices_u16,
  partition_slices_i8,
  partition_slices_u8,
} from './bins/partition.wasm';
import { ensureMemory, resetAllocator, copyIn, copyOut, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type PartitionFn = (aPtr: number, N: number, kth: number) => void;
type SlicePartitionFn = (aPtr: number, sliceSize: number, numSlices: number, kth: number) => void;

const kernels: Partial<Record<DType, PartitionFn>> = {
  float64: partition_f64,
  float32: partition_f32,
  int64: partition_i64,
  uint64: partition_u64,
  int32: partition_i32,
  uint32: partition_u32,
  int16: partition_i16,
  uint16: partition_u16,
  int8: partition_i8,
  uint8: partition_u8,
};

const sliceKernels: Partial<Record<DType, SlicePartitionFn>> = {
  float64: partition_slices_f64,
  float32: partition_slices_f32,
  int64: partition_slices_i64,
  uint64: partition_slices_u64,
  int32: partition_slices_i32,
  uint32: partition_slices_u32,
  int16: partition_slices_i16,
  uint16: partition_slices_u16,
  int8: partition_slices_i8,
  uint8: partition_slices_u8,
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
 * WASM-accelerated partition of contiguous slices.
 * Uses batch kernel when slices are packed contiguously.
 */
export function wasmPartitionSlices(
  resultData: TypedArray,
  sliceOffsets: Int32Array | number[],
  axisSize: number,
  outerSize: number,
  kth: number,
  dtype: DType
): boolean {
  if (axisSize < 2) return true;

  const sliceKernel = sliceKernels[dtype];

  if (sliceKernel && sliceOffsets[0] === 0 && outerSize > 1 && sliceOffsets[1] === axisSize) {
    const Ctor = ctorMap[dtype];
    if (!Ctor) return false;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const totalBytes = resultData.length * bpe;

    ensureMemory(totalBytes);
    resetAllocator();

    const ptr = copyIn(resultData as TypedArray);
    sliceKernel(ptr, axisSize, outerSize, kth);

    const mem = getSharedMemory();
    new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
      new Uint8Array(mem.buffer, ptr, resultData.byteLength)
    );
    return true;
  }

  // Fallback: per-slice calls
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return false;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalBytes = resultData.length * bpe;

  ensureMemory(totalBytes);
  resetAllocator();

  const ptr = copyIn(resultData as TypedArray);

  for (let i = 0; i < outerSize; i++) {
    kernel(ptr + sliceOffsets[i]! * bpe, axisSize, kth);
  }

  const mem = getSharedMemory();
  new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
    new Uint8Array(mem.buffer, ptr, resultData.byteLength)
  );

  return true;
}

/**
 * WASM-accelerated partition of a contiguous 1D buffer.
 * Returns partitioned ArrayStorage or null if WASM can't handle it.
 */
export function wasmPartition(a: ArrayStorage, kth: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;

  ensureMemory(aBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);

  kernel(aPtr, size, kth);

  const outData = copyOut(
    aPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
