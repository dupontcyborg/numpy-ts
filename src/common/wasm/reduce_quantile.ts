/**
 * WASM-accelerated quantile computation.
 *
 * Sorts a copy of the data in-place using native-dtype sort kernels (same as
 * wasmSort), then interpolates. No f64 conversion — operates on the original
 * dtype for both sort and readback.
 *
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
} from './bins/sort.wasm';
import {
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  getSharedMemory,
  wasmMalloc,
} from './runtime';
import { ArrayStorage } from '../storage';
import {
  effectiveDType,
  isComplexDType,
  getDTypeSize,
  type DType,
  type TypedArray,
} from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type SortFn = (aPtr: number, N: number) => void;
type SortWithScratchFn = (aPtr: number, N: number, scratchPtr: number) => void;
type SliceSortFn = (aPtr: number, sliceSize: number, numSlices: number) => void;

const sortKernels: Partial<Record<DType, SortFn>> = {
  float64: sort_f64,
  float32: sort_f32,
  float16: sort_f16,
  int64: sort_i64,
  uint64: sort_u64,
  int32: sort_i32,
  uint32: sort_u32,
};

// Narrow integer types accept an optional scratch pointer for radix sort
const sortWithScratchKernels: Partial<Record<DType, SortWithScratchFn>> = {
  int16: sort_i16 as unknown as SortWithScratchFn,
  uint16: sort_u16 as unknown as SortWithScratchFn,
  int8: sort_i8 as unknown as SortWithScratchFn,
  uint8: sort_u8 as unknown as SortWithScratchFn,
};

const sliceSortKernels: Partial<Record<DType, SliceSortFn>> = {
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
};

/**
 * Read a numeric value from a sorted WASM buffer at a given element index.
 */
function readSorted(mem: ArrayBuffer, ptr: number, idx: number, dtype: DType): number {
  const bpe = getDTypeSize(dtype);
  const byteOff = ptr + idx * bpe;
  switch (dtype) {
    case 'float64':
      return new Float64Array(mem, byteOff, 1)[0]!;
    case 'float32':
      return new Float32Array(mem, byteOff, 1)[0]!;
    case 'float16':
      return new Float16Array(mem, byteOff, 1)[0]! as number;
    case 'int64':
      return Number(new BigInt64Array(mem, byteOff, 1)[0]!);
    case 'uint64':
      return Number(new BigUint64Array(mem, byteOff, 1)[0]!);
    case 'int32':
      return new Int32Array(mem, byteOff, 1)[0]!;
    case 'uint32':
      return new Uint32Array(mem, byteOff, 1)[0]!;
    case 'int16':
      return new Int16Array(mem, byteOff, 1)[0]!;
    case 'uint16':
      return new Uint16Array(mem, byteOff, 1)[0]!;
    case 'int8':
      return new Int8Array(mem, byteOff, 1)[0]!;
    case 'uint8':
      return new Uint8Array(mem, byteOff, 1)[0]!;
    default:
      return 0;
  }
}

/**
 * Interpolate quantile from a sorted buffer in WASM memory.
 */
function interpolateQuantile(
  mem: ArrayBuffer,
  ptr: number,
  N: number,
  q: number,
  dtype: DType
): number {
  const idx = q * (N - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return readSorted(mem, ptr, lower, dtype);
  const frac = idx - lower;
  const lo = readSorted(mem, ptr, lower, dtype);
  const hi = readSorted(mem, ptr, upper, dtype);
  return lo * (1 - frac) + hi * frac;
}

/**
 * WASM-accelerated quantile (no axis, full array).
 * Sorts a scratch copy in the native dtype, then interpolates.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmReduceQuantile(a: ArrayStorage, q: number): number | null {
  if (!a.isCContiguous) return null;
  const dtype = effectiveDType(a.dtype);
  if (isComplexDType(dtype)) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const kernel = sortKernels[dtype];
  const scratchKernel = sortWithScratchKernels[dtype];
  if (!kernel && !scratchKernel) return null;

  const bpe = getDTypeSize(dtype);

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Copy input data to scratch in its native dtype
  const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  // Sort needs a mutable copy — if resolveInputPtr returned a direct pointer
  // (WASM-backed, zero-copy), we need to copy to scratch to avoid mutating the original
  let sortPtr: number;
  if (a.isWasmBacked && a.offset === 0) {
    // Direct pointer — copy to scratch for in-place sort
    const copyBytes = size * bpe;
    sortPtr = scratchAlloc(copyBytes);
    const mem = getSharedMemory();
    new Uint8Array(mem.buffer, sortPtr, copyBytes).set(
      new Uint8Array(mem.buffer, inPtr, copyBytes)
    );
  } else {
    // Already copied to scratch by resolveInputPtr
    sortPtr = inPtr;
  }

  // Sort in-place in native dtype — allocate radix scratch for narrow types
  if (scratchKernel) {
    const radixScratch = scratchAlloc(size * bpe);
    scratchKernel(sortPtr, size, radixScratch);
  } else {
    kernel!(sortPtr, size);
  }

  // Read quantile from sorted buffer
  const mem = getSharedMemory();
  return interpolateQuantile(mem.buffer, sortPtr, size, q, dtype);
}

/**
 * WASM-accelerated strided quantile along an axis.
 * For shape [outer, axisSize, inner], sorts each column and interpolates.
 * Output is [outer * inner] f64, or null if WASM can't handle.
 */
export function wasmReduceQuantileStrided(
  storage: ArrayStorage,
  outer: number,
  axisSize: number,
  inner: number,
  q: number
): ArrayStorage | null {
  if (!storage.isCContiguous) return null;
  const dtype = effectiveDType(storage.dtype);
  if (isComplexDType(dtype)) return null;
  const totalSize = outer * axisSize * inner;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const sliceKernel = sliceSortKernels[dtype];
  const sortKernel = sortKernels[dtype];
  const scratchSortKernel = sortWithScratchKernels[dtype];
  if (!sortKernel && !scratchSortKernel) return null;

  const bpe = getDTypeSize(dtype);
  const outSize = outer * inner;
  const outBytes = outSize * 8; // output is always f64

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Copy input to scratch for in-place sorting
  const inPtr = resolveInputPtr(
    storage.data,
    storage.isWasmBacked,
    storage.wasmPtr,
    storage.offset,
    totalSize,
    bpe
  );

  let sortPtr: number;
  if (storage.isWasmBacked && storage.offset === 0) {
    const copyBytes = totalSize * bpe;
    sortPtr = scratchAlloc(copyBytes);
    const mem = getSharedMemory();
    new Uint8Array(mem.buffer, sortPtr, copyBytes).set(
      new Uint8Array(mem.buffer, inPtr, copyBytes)
    );
  } else {
    sortPtr = inPtr;
  }

  const mem = getSharedMemory();
  const outView = new Float64Array(mem.buffer, outRegion.ptr, outSize);

  if (inner === 1 && sliceKernel) {
    // Contiguous slices along last axis — batch sort
    sliceKernel(sortPtr, axisSize, outer);

    // Read quantile from each sorted slice
    for (let o = 0; o < outer; o++) {
      const slicePtr = sortPtr + o * axisSize * bpe;
      outView[o] = interpolateQuantile(mem.buffer, slicePtr, axisSize, q, dtype);
    }
  } else {
    // Non-contiguous: gather each column into a temp buffer, sort, interpolate
    const colBytes = axisSize * bpe;
    const colPtr = scratchAlloc(colBytes);
    const stride = axisSize * inner;

    for (let o = 0; o < outer; o++) {
      for (let inn = 0; inn < inner; inn++) {
        // Gather column values
        const base = sortPtr + (o * stride + inn) * bpe;
        const colView = new Uint8Array(mem.buffer, colPtr, colBytes);
        for (let k = 0; k < axisSize; k++) {
          const srcOff = base + k * inner * bpe;
          colView.set(new Uint8Array(mem.buffer, srcOff, bpe), k * bpe);
        }

        // Sort the column
        if (scratchSortKernel) {
          const colScratch = scratchAlloc(axisSize * bpe);
          scratchSortKernel(colPtr, axisSize, colScratch);
        } else {
          sortKernel!(colPtr, axisSize);
        }

        // Interpolate
        outView[o * inner + inn] = interpolateQuantile(mem.buffer, colPtr, axisSize, q, dtype);
      }
    }
  }

  return ArrayStorage.fromWasmRegion(
    [outSize],
    'float64',
    outRegion,
    outSize,
    Float64Array as unknown as new (
      buf: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
