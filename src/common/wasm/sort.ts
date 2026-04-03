/**
 * WASM-accelerated in-place array sorting.
 *
 * Sorts a contiguous 1D buffer in-place using heap sort.
 * Returns null if WASM can't handle this case.
 */

import {
  sort_f64,
  sort_f32,
  sort_f16,
  sort_i64,
  sort_u64,
  sort_i32,
  sort_u32,
  sort_i16,
  sort_u16,
  sort_i8,
  sort_u8,
  sort_slices_f64,
  sort_slices_f32,
  sort_slices_f16,
  sort_slices_i64,
  sort_slices_u64,
  sort_slices_i32,
  sort_slices_u32,
  sort_slices_i16,
  sort_slices_u16,
  sort_slices_i8,
  sort_slices_u8,
  sort_c128,
  sort_c64,
  sort_slices_c128,
  sort_slices_c64,
} from './bins/sort.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  scratchAlloc,
  scratchCopyIn,
  getSharedMemory,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type SortFn = (aPtr: number, N: number) => void;
type SliceSortFn = (aPtr: number, sliceSize: number, numSlices: number) => void;

type SortWithScratchFn = (aPtr: number, N: number, scratchPtr: number) => void;

const kernels: Partial<Record<DType, SortFn>> = {
  float64: sort_f64,
  float32: sort_f32,
  float16: sort_f16,
  int64: sort_i64,
  uint64: sort_u64,
  int32: sort_i32,
  uint32: sort_u32,
  complex128: sort_c128,
  complex64: sort_c64,
};

// Narrow integer types accept scratch pointer for radix sort at any N
const scratchKernels: Partial<Record<DType, SortWithScratchFn>> = {
  int16: sort_i16 as unknown as SortWithScratchFn,
  uint16: sort_u16 as unknown as SortWithScratchFn,
  int8: sort_i8 as unknown as SortWithScratchFn,
  uint8: sort_u8 as unknown as SortWithScratchFn,
};

const sliceKernels: Partial<Record<DType, SliceSortFn>> = {
  float64: sort_slices_f64,
  float32: sort_slices_f32,
  float16: sort_slices_f16,
  int64: sort_slices_i64,
  uint64: sort_slices_u64,
  int32: sort_slices_i32,
  uint32: sort_slices_u32,
  int16: sort_slices_i16,
  uint16: sort_slices_u16,
  int8: sort_slices_i8,
  uint8: sort_slices_u8,
  complex128: sort_slices_c128,
  complex64: sort_slices_c64,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Uint16Array, // native f16 sort operates on raw u16 bits
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
  complex128: Float64Array,
  complex64: Float32Array,
};

/**
 * WASM-accelerated sort of contiguous slices in a typed array buffer.
 * Uses a single batched WASM call for all slices (eliminates per-slice JS→WASM overhead).
 * Returns true if WASM handled it.
 *
 * Note: wasmSortSlices operates on pre-existing JS buffers (resultData), so it uses
 * scratch copy-in and copy-out rather than wasmMalloc. The caller owns the buffer.
 */
export function wasmSortSlices(
  resultData: TypedArray,
  sliceOffsets: Int32Array | number[],
  axisSize: number,
  outerSize: number,
  dtype: DType
): boolean {
  if (axisSize < 2) return true;

  const sliceKernel = sliceKernels[dtype];

  if (sliceKernel && sliceOffsets[0] === 0 && outerSize > 1 && sliceOffsets[1] === axisSize) {
    // Packed contiguous slices — single batch WASM call
    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const ptr = scratchCopyIn(resultData);
    sliceKernel(ptr, axisSize, outerSize);

    const mem = getSharedMemory();
    new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
      new Uint8Array(mem.buffer, ptr, resultData.byteLength)
    );
    return true;
  }

  // Non-contiguous slices: per-slice calls
  const kernel = kernels[dtype];
  const scratchKernel = scratchKernels[dtype];
  const Ctor = ctorMap[dtype];
  if ((!kernel && !scratchKernel) || !Ctor) return false;

  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const bytesPerElem = isComplex ? bpe * 2 : bpe;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const ptr = scratchCopyIn(resultData);

  if (scratchKernel) {
    const radixScratch = scratchAlloc(axisSize * bpe);
    for (let i = 0; i < outerSize; i++) {
      scratchKernel(ptr + sliceOffsets[i]! * bytesPerElem, axisSize, radixScratch);
    }
  } else {
    for (let i = 0; i < outerSize; i++) {
      kernel!(ptr + sliceOffsets[i]! * bytesPerElem, axisSize);
    }
  }

  const mem = getSharedMemory();
  new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
    new Uint8Array(mem.buffer, ptr, resultData.byteLength)
  );

  return true;
}

/**
 * WASM-accelerated sort of a contiguous 1D buffer.
 * Returns sorted ArrayStorage or null if WASM can't handle it.
 */
export function wasmSort(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const scratchK = scratchKernels[dtype];
  const Ctor = ctorMap[dtype];
  if ((!kernel && !scratchK) || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const isF16 = dtype === 'float16';
  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  const bufLen = isComplex ? size * 2 : size;
  const outBytes = bufLen * bpe;

  // Sort is in-place: we allocate output, copy input into it, sort in-place
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aOff = a.offset;
  if (isF16) {
    const scratchPtr = f16InputToScratchF32(a, size);
    kernel!(scratchPtr, size);
    // Copy sorted f32 scratch into persistent output, then convert to f16
    const mem = getSharedMemory();
    new Uint8Array(mem.buffer, outRegion.ptr, outBytes).set(
      new Uint8Array(mem.buffer, scratchPtr, outBytes)
    );
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      f16Region,
      size,
      Uint16Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
  }

  // Copy input data into the output region, then sort in-place
  const mem = getSharedMemory();
  if (a.isWasmBacked) {
    new Uint8Array(mem.buffer, outRegion.ptr, outBytes).set(
      new Uint8Array(mem.buffer, a.wasmPtr + aOff * bpe, outBytes)
    );
  } else {
    const aData = a.data.subarray(aOff, aOff + bufLen) as TypedArray;
    new Uint8Array(mem.buffer, outRegion.ptr, outBytes).set(
      new Uint8Array(aData.buffer, aData.byteOffset, aData.byteLength)
    );
  }

  if (scratchK) {
    const radixScratch = scratchAlloc(size * bpe);
    scratchK(outRegion.ptr, size, radixScratch);
  } else {
    kernel!(outRegion.ptr, size);
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    bufLen,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
