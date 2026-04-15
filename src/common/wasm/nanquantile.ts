/**
 * WASM-accelerated nanquantile computation.
 *
 * Filters NaN values, sorts the remaining, and computes quantile via
 * linear interpolation — all inside the WASM kernel. Float-only.
 *
 * Returns null if WASM can't handle this case.
 */

import {
  nanquantile_f64,
  nanquantile_f32,
  nanquantile_strided_f64,
  nanquantile_strided_f32,
} from './bins/nanquantile.wasm';
import { resetScratchAllocator, resolveInputPtr, scratchAlloc, wasmMalloc } from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type FlatKernel = (aPtr: number, N: number, q: number, workPtr: number) => number;
type StridedKernel = (
  aPtr: number,
  outPtr: number,
  outer: number,
  axisSize: number,
  inner: number,
  q: number,
  workPtr: number
) => void;

const flatKernels: Partial<Record<DType, FlatKernel>> = {
  float64: nanquantile_f64,
  float32: nanquantile_f32,
};

const stridedKernels: Partial<Record<DType, StridedKernel>> = {
  float64: nanquantile_strided_f64,
  float32: nanquantile_strided_f32,
};

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
};

/**
 * WASM-accelerated nanquantile (no axis, full array).
 * Returns null if WASM can't handle (complex, non-contiguous, too small, non-float).
 */
export function wasmNanquantile(a: ArrayStorage, q: number): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = flatKernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  // Allocate work buffer (size elements in native dtype)
  const workPtr = scratchAlloc(size * bpe);

  return kernel(aPtr, size, q, workPtr);
}

/**
 * WASM-accelerated strided nanquantile along an axis.
 * For shape [outer, axisSize, inner], filters NaN per column, sorts, interpolates.
 * Output is [outer * inner] f64, or null if WASM can't handle.
 */
export function wasmNanquantileStrided(
  storage: ArrayStorage,
  outer: number,
  axisSize: number,
  inner: number,
  q: number
): ArrayStorage | null {
  if (!storage.isCContiguous) return null;

  const dtype = effectiveDType(storage.dtype);
  const totalSize = outer * axisSize * inner;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const kernel = stridedKernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  const outSize = outer * inner;
  const outBytes = outSize * 8; // output is always f64

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(
    storage.data,
    storage.isWasmBacked,
    storage.wasmPtr,
    storage.offset,
    totalSize,
    bpe
  );

  // Allocate work buffer (axis_size elements in native dtype)
  const workPtr = scratchAlloc(axisSize * bpe);

  kernel(aPtr, outRegion.ptr, outer, axisSize, inner, q, workPtr);

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
