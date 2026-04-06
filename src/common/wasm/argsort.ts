/**
 * WASM-accelerated argsort (returns indices that would sort an array).
 *
 * Returns null if WASM can't handle this case.
 */

import {
  argsort_f64,
  argsort_f32,
  argsort_i64,
  argsort_u64,
  argsort_i32,
  argsort_u32,
  argsort_i16,
  argsort_u16,
  argsort_i8,
  argsort_u8,
  argsort_slices_f64,
  argsort_slices_f32,
  argsort_slices_i64,
  argsort_slices_u64,
  argsort_slices_i32,
  argsort_slices_u32,
  argsort_slices_i16,
  argsort_slices_u16,
  argsort_slices_i8,
  argsort_slices_u8,
  argsort_c128,
  argsort_c64,
  argsort_slices_c128,
  argsort_slices_c64,
} from './bins/argsort.wasm';
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
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ArgsortFn = (aPtr: number, outPtr: number, N: number) => void;
type SliceArgsortFn = (aPtr: number, outPtr: number, sliceSize: number, numSlices: number) => void;

const kernels: Partial<Record<DType, ArgsortFn>> = {
  float64: argsort_f64,
  float32: argsort_f32,
  int64: argsort_i64,
  uint64: argsort_u64,
  int32: argsort_i32,
  uint32: argsort_u32,
  int16: argsort_i16,
  uint16: argsort_u16,
  int8: argsort_i8,
  uint8: argsort_u8,
  complex128: argsort_c128,
  complex64: argsort_c64,
  float16: argsort_f32,
};

const sliceKernels: Partial<Record<DType, SliceArgsortFn>> = {
  float64: argsort_slices_f64,
  float32: argsort_slices_f32,
  int64: argsort_slices_i64,
  uint64: argsort_slices_u64,
  int32: argsort_slices_i32,
  uint32: argsort_slices_u32,
  int16: argsort_slices_i16,
  uint16: argsort_slices_u16,
  int8: argsort_slices_i8,
  uint8: argsort_slices_u8,
  complex128: argsort_slices_c128,
  complex64: argsort_slices_c64,
  float16: argsort_slices_f32,
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
  float16: Float32Array,
};

/**
 * WASM-accelerated argsort of contiguous slices.
 * Uses batch kernel when slices are packed contiguously.
 *
 * Note: operates on pre-existing JS buffers (inputData/resultData), uses scratch.
 */
export function wasmArgsortSlices(
  inputData: TypedArray,
  resultData: Int32Array,
  inputSliceOffsets: Int32Array | number[],
  outputSliceOffsets: Int32Array | number[],
  axisSize: number,
  outerSize: number,
  dtype: DType
): boolean {
  if (axisSize < 2) return false;

  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  const sliceKernel = sliceKernels[dtype];

  // Try batch kernel for packed contiguous slices
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

    sliceKernel(inputPtr, outputPtr, axisSize, outerSize);

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
  const bytesPerElem = isComplex ? bpe * 2 : bpe;
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
      inputPtr + inputSliceOffsets[i]! * bytesPerElem,
      outputPtr + outputSliceOffsets[i]! * 4,
      axisSize
    );
  }

  const mem = getSharedMemory();
  new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength).set(
    new Uint8Array(mem.buffer, outputPtr, resultData.byteLength)
  );
  return true;
}

/**
 * WASM-accelerated argsort of a contiguous 1D buffer.
 * Returns ArrayStorage of int32 indices or null if WASM can't handle it.
 */
export function wasmArgsort(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const isComplex = dtype === 'complex128' || dtype === 'complex64';
  const bufLen = isComplex ? size * 2 : size;
  const outBytes = size * 4; // i32 indices

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aPtr =
    dtype === 'float16'
      ? f16InputToScratchF32(a, size)
      : resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, bufLen, bpe);

  kernel(aPtr, outRegion.ptr, size);

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
