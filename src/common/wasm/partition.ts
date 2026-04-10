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
import {
  wasmMalloc,
  resetScratchAllocator,
  scratchCopyIn,
  getSharedMemory,
  f16InputToScratchF32,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, hasFloat16, type DType, TypedArray } from '../dtype';
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
  float16: partition_f32,
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
  float16: partition_slices_f32,
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
  float16: Float32Array,
};

/**
 * WASM-accelerated partition of contiguous slices.
 * Uses batch kernel when slices are packed contiguously.
 *
 * Note: operates on pre-existing JS buffers, uses scratch.
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
  const isF16 = dtype === 'float16';

  if (sliceKernel && sliceOffsets[0] === 0 && outerSize > 1 && sliceOffsets[1] === axisSize) {
    const Ctor = ctorMap[dtype];
    if (!Ctor) return false;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const ptr = isF16
      ? f16InputToScratchF32(
          { data: resultData, isWasmBacked: false, wasmPtr: 0, offset: 0 },
          resultData.length
        )
      : scratchCopyIn(resultData as TypedArray);
    sliceKernel(ptr, axisSize, outerSize, kth);

    const mem = getSharedMemory();
    if (isF16) {
      const f16Out = new Uint16Array(resultData.length);
      const f32View = new Float32Array(mem.buffer, ptr, resultData.length);
      new Float16Array(f16Out.buffer, 0, resultData.length).set(f32View);
      new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
        new Uint8Array(f16Out.buffer, 0, f16Out.byteLength)
      );
    } else {
      new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
        new Uint8Array(mem.buffer, ptr, resultData.byteLength)
      );
    }
    return true;
  }

  // Fallback: per-slice calls
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return false;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const ptr = isF16
    ? f16InputToScratchF32(
        { data: resultData, isWasmBacked: false, wasmPtr: 0, offset: 0 },
        resultData.length
      )
    : scratchCopyIn(resultData as TypedArray);

  for (let i = 0; i < outerSize; i++) {
    kernel(ptr + sliceOffsets[i]! * bpe, axisSize, kth);
  }

  const mem = getSharedMemory();
  if (isF16) {
    const f32View = new Float32Array(mem.buffer, ptr, resultData.length);
    if (hasFloat16) {
      const f16Out = new Uint16Array(resultData.length);
      new Float16Array(f16Out.buffer, 0, resultData.length).set(f32View);
      new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
        new Uint8Array(f16Out.buffer, 0, f16Out.byteLength)
      );
    } else {
      // Polyfill: float16 arrays are backed by Float32Array, copy f32 values directly
      (resultData as Float32Array).set(f32View);
    }
  } else {
    new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
      new Uint8Array(mem.buffer, ptr, resultData.byteLength)
    );
  }

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

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;
  const isF16 = dtype === 'float16';

  // Partition is in-place: allocate output, copy input, partition in-place
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aOff = a.offset;
  if (isF16) {
    // Float16 partition: use the "sort floats as integers" trick.
    // Partition only rearranges elements — it doesn't modify values — so we can
    // transform f16 bit patterns into a form where i16 comparison gives the
    // correct f16 order, run partition_i16, and undo the transform.
    //
    // Transform: if sign bit is set (negative), XOR with 0x7FFF (flip exp+mantissa).
    // This makes signed i16 comparison match f16 ordering.
    // The transform is self-inverse: applying it again undoes it.
    const f16Bytes = size * 2;
    const f16Region = wasmMalloc(f16Bytes);
    if (!f16Region) {
      outRegion.release();
      return null;
    }
    const mem = getSharedMemory();

    // Copy f16 input into f16Region
    if (a.isWasmBacked) {
      new Uint8Array(mem.buffer, f16Region.ptr, f16Bytes).set(
        new Uint8Array(mem.buffer, a.wasmPtr + aOff * 2, f16Bytes)
      );
    } else {
      const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
      new Uint8Array(mem.buffer, f16Region.ptr, f16Bytes).set(
        new Uint8Array(aData.buffer, aData.byteOffset, aData.byteLength)
      );
    }

    // Transform: make i16 comparison match f16 order
    const u16View = new Uint16Array(mem.buffer, f16Region.ptr, size);
    for (let j = 0; j < size; j++) {
      if (u16View[j]! & 0x8000) u16View[j]! ^= 0x7fff;
    }

    // Partition using i16 kernel
    partition_i16(f16Region.ptr, size, kth);

    // Undo transform
    for (let j = 0; j < size; j++) {
      if (u16View[j]! & 0x8000) u16View[j]! ^= 0x7fff;
    }

    outRegion.release();
    const F16Ctor = hasFloat16
      ? (Float16Array as unknown as new (
          buffer: ArrayBuffer,
          byteOffset: number,
          length: number
        ) => TypedArray)
      : (Uint16Array as unknown as new (
          buffer: ArrayBuffer,
          byteOffset: number,
          length: number
        ) => TypedArray);
    return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, f16Region, size, F16Ctor);
  }

  // Copy input into output region, then partition in-place
  const mem = getSharedMemory();
  if (a.isWasmBacked) {
    new Uint8Array(mem.buffer, outRegion.ptr, outBytes).set(
      new Uint8Array(mem.buffer, a.wasmPtr + aOff * bpe, outBytes)
    );
  } else {
    const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
    new Uint8Array(mem.buffer, outRegion.ptr, outBytes).set(
      new Uint8Array(aData.buffer, aData.byteOffset, aData.byteLength)
    );
  }

  kernel(outRegion.ptr, size, kth);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
