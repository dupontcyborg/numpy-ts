/**
 * WASM-accelerated in-place array sorting.
 *
 * Sorts a contiguous 1D buffer in-place using heap sort.
 * Returns null if WASM can't handle this case.
 */

import {
  sort_f64,
  sort_f32,
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
import { ensureMemory, resetAllocator, copyIn, copyOut, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type SortFn = (aPtr: number, N: number) => void;
type SliceSortFn = (aPtr: number, sliceSize: number, numSlices: number) => void;

const kernels: Partial<Record<DType, SortFn>> = {
  float64: sort_f64,
  float32: sort_f32,
  int64: sort_i64,
  uint64: sort_u64,
  int32: sort_i32,
  uint32: sort_u32,
  int16: sort_i16,
  uint16: sort_u16,
  int8: sort_i8,
  uint8: sort_u8,
  complex128: sort_c128,
  complex64: sort_c64,
};

const sliceKernels: Partial<Record<DType, SliceSortFn>> = {
  float64: sort_slices_f64,
  float32: sort_slices_f32,
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
 */
export function wasmSortSlices(
  resultData: TypedArray,
  sliceOffsets: Int32Array | number[],
  axisSize: number,
  outerSize: number,
  dtype: DType
): boolean {
  if (axisSize < 2) return true;

  // Check if slices are packed contiguously (last-axis sort on C-contiguous array)
  // If so, use the batch kernel (one WASM call). Otherwise fall back to per-slice calls.
  // sliceOffsets are in logical element units (complex = 1 element, not 2 floats).
  const sliceKernel = sliceKernels[dtype];

  if (sliceKernel && sliceOffsets[0] === 0 && outerSize > 1 &&
      sliceOffsets[1] === axisSize) {
    // Packed contiguous slices — single batch WASM call
    const Ctor = ctorMap[dtype];
    if (!Ctor) return false;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const totalBytes = resultData.length * bpe;

    ensureMemory(totalBytes);
    resetAllocator();

    const ptr = copyIn(resultData as TypedArray);
    sliceKernel(ptr, axisSize, outerSize);

    const mem = getSharedMemory();
    new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
      new Uint8Array(mem.buffer, ptr, resultData.byteLength)
    );
    return true;
  }

  // Non-contiguous slices: per-slice calls
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return false;

  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  // For complex, each logical element is 2 floats, so byte offset = logicalOffset * 2 * bpe
  const bytesPerElem = isComplex ? bpe * 2 : bpe;
  const totalBytes = resultData.length * bpe;

  ensureMemory(totalBytes);
  resetAllocator();

  const ptr = copyIn(resultData as TypedArray);

  for (let i = 0; i < outerSize; i++) {
    kernel(ptr + sliceOffsets[i]! * bytesPerElem, axisSize);
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
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;

  ensureMemory(aBytes);
  resetAllocator();

  const aOff = a.offset;
  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  const bufLen = isComplex ? size * 2 : size;
  const aData = a.data.subarray(aOff, aOff + bufLen) as TypedArray;

  const aPtr = copyIn(aData);

  kernel(aPtr, size);

  const outData = copyOut(
    aPtr,
    bufLen,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
