/**
 * WASM-accelerated argpartition (returns indices for quickselect partition).
 *
 * Returns null if WASM can't handle this case.
 */

import {
  argpartition_f64,
  argpartition_f32,
  argpartition_i64,
  argpartition_u64,
  argpartition_i32,
  argpartition_u32,
  argpartition_i16,
  argpartition_u16,
  argpartition_i8,
  argpartition_u8,
  argpartition_slices_f64,
  argpartition_slices_f32,
  argpartition_slices_i64,
  argpartition_slices_u64,
  argpartition_slices_i32,
  argpartition_slices_u32,
  argpartition_slices_i16,
  argpartition_slices_u16,
  argpartition_slices_i8,
  argpartition_slices_u8,
} from './bins/argpartition.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchCopyIn,
  scratchAlloc,
  getSharedMemory,
  f16InputToScratchF32,
} from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ArgpartitionFn = (aPtr: number, outPtr: number, N: number, kth: number) => void;
type SliceArgpartitionFn = (
  aPtr: number,
  outPtr: number,
  sliceSize: number,
  numSlices: number,
  kth: number
) => void;

const kernels: Partial<Record<DType, ArgpartitionFn>> = {
  float64: argpartition_f64,
  float32: argpartition_f32,
  int64: argpartition_i64,
  uint64: argpartition_u64,
  int32: argpartition_i32,
  uint32: argpartition_u32,
  int16: argpartition_i16,
  uint16: argpartition_u16,
  int8: argpartition_i8,
  uint8: argpartition_u8,
  float16: argpartition_f32,
};

const sliceKernels: Partial<Record<DType, SliceArgpartitionFn>> = {
  float64: argpartition_slices_f64,
  float32: argpartition_slices_f32,
  int64: argpartition_slices_i64,
  uint64: argpartition_slices_u64,
  int32: argpartition_slices_i32,
  uint32: argpartition_slices_u32,
  int16: argpartition_slices_i16,
  uint16: argpartition_slices_u16,
  int8: argpartition_slices_i8,
  uint8: argpartition_slices_u8,
  float16: argpartition_slices_f32,
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
 * WASM-accelerated argpartition of contiguous slices.
 * Uses batch kernel when slices are packed contiguously.
 *
 * Note: operates on pre-existing JS buffers, uses scratch.
 */
export function wasmArgpartitionSlices(
  inputData: TypedArray,
  resultData: Int32Array,
  inputSliceOffsets: Int32Array | number[],
  outputSliceOffsets: Int32Array | number[],
  axisSize: number,
  outerSize: number,
  kth: number,
  dtype: DType
): boolean {
  if (axisSize < 2) return false;

  const sliceKernel = sliceKernels[dtype];

  if (
    sliceKernel &&
    inputSliceOffsets[0] === 0 &&
    outerSize > 1 &&
    inputSliceOffsets[1] === axisSize &&
    outputSliceOffsets[0] === 0 &&
    outputSliceOffsets[1] === axisSize
  ) {
    const Ctor = ctorMap[dtype];
    if (!Ctor) return false;
    const outputBytes = resultData.length * 4;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const inputPtr =
      dtype === 'float16'
        ? f16InputToScratchF32(
            { data: inputData, isWasmBacked: false, wasmPtr: 0, offset: 0 },
            inputData.length
          )
        : scratchCopyIn(inputData as TypedArray);
    const outputPtr = scratchAlloc(outputBytes);

    sliceKernel(inputPtr, outputPtr, axisSize, outerSize, kth);

    const mem = getSharedMemory();
    new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
      new Uint8Array(mem.buffer, outputPtr, resultData.byteLength)
    );
    return true;
  }

  // Fallback: per-slice calls
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return false;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outputBytes = resultData.length * 4;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inputPtr =
    dtype === 'float16'
      ? f16InputToScratchF32(
          { data: inputData, isWasmBacked: false, wasmPtr: 0, offset: 0 },
          inputData.length
        )
      : scratchCopyIn(inputData as TypedArray);
  const outputPtr = scratchAlloc(outputBytes);

  for (let i = 0; i < outerSize; i++) {
    kernel(
      inputPtr + inputSliceOffsets[i]! * bpe,
      outputPtr + outputSliceOffsets[i]! * 4,
      axisSize,
      kth
    );
  }

  const mem = getSharedMemory();
  new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
    new Uint8Array(mem.buffer, outputPtr, resultData.byteLength)
  );

  return true;
}

/**
 * WASM-accelerated argpartition of a contiguous 1D buffer.
 * Returns ArrayStorage of int32 indices or null if WASM can't handle it.
 */
export function wasmArgpartition(a: ArrayStorage, kth: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const outBytes = size * 4; // i32 indices

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  let aPtr: number;
  let actualKernel = kernel;

  if (dtype === 'float16') {
    // "Sort floats as integers" trick: transform f16 bit patterns so that
    // signed i16 comparison matches f16 ordering, then use argpartition_i16.
    const f16Bytes = size * 2;
    const mem = getSharedMemory();
    aPtr = scratchAlloc(f16Bytes);
    if (a.isWasmBacked) {
      new Uint8Array(mem.buffer, aPtr, f16Bytes).set(
        new Uint8Array(mem.buffer, a.wasmPtr + a.offset * 2, f16Bytes)
      );
    } else {
      const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
      new Uint8Array(mem.buffer, aPtr, f16Bytes).set(
        new Uint8Array(aData.buffer, aData.byteOffset, aData.byteLength)
      );
    }
    const u16View = new Uint16Array(mem.buffer, aPtr, size);
    for (let j = 0; j < size; j++) {
      if (u16View[j]! & 0x8000) u16View[j]! ^= 0x7fff;
    }
    actualKernel = argpartition_i16;
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  }

  actualKernel(aPtr, outRegion.ptr, size, kth);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'int32',
    outRegion,
    size,
    Int32Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
