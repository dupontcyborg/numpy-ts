/**
 * WASM-accelerated quantile computation.
 *
 * Sorts a copy of the data in WASM, then interpolates.
 * All dtypes are converted to f64 before passing to WASM.
 * Returns null if WASM can't handle this case.
 */

import { reduce_quantile_f64, reduce_quantile_strided_f64 } from './bins/reduce_quantile.wasm';
import {
  resetScratchAllocator,
  scratchCopyIn,
  scratchAlloc,
  wasmMalloc,
  f16ToF32Input,
} from './runtime';
import { ArrayStorage } from '../storage';
import { isBigIntDType, isComplexDType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

/**
 * WASM-accelerated quantile (no axis, full array).
 * Converts all data to f64, sorts in WASM, interpolates.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmReduceQuantile(a: ArrayStorage, q: number): number | null {
  if (!a.isCContiguous) return null;
  if (isComplexDType(a.dtype)) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Convert to f64 in JS, then copy to WASM
  const f64Buf = new Float64Array(size);
  const off = a.offset;
  const data = a.data;

  if (a.dtype === 'float64') {
    f64Buf.set((data as Float64Array).subarray(off, off + size));
  } else if (a.dtype === 'float16') {
    const f32Data = f16ToF32Input(
      data.subarray(off, off + size) as TypedArray,
      a.dtype
    ) as Float32Array;
    for (let i = 0; i < size; i++) f64Buf[i] = f32Data[i]!;
  } else if (isBigIntDType(a.dtype)) {
    const typed = data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) f64Buf[i] = Number(typed[off + i]!);
  } else {
    for (let i = 0; i < size; i++) f64Buf[i] = Number(data[off + i]!);
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const ptr = scratchCopyIn(f64Buf as unknown as TypedArray);

  // Sort in-place in WASM and compute quantile
  return reduce_quantile_f64(ptr, size, q);
}

/**
 * WASM-accelerated strided quantile along an axis.
 * For shape [outer, axisSize, inner], computes quantile along the axis dimension.
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
  if (isComplexDType(storage.dtype)) return null;

  const totalSize = outer * axisSize * inner;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const outSize = outer * inner;
  const outBytes = outSize * 8; // f64

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Convert input to f64 in JS, then copy to scratch
  const f64Buf = new Float64Array(totalSize);
  const off = storage.offset;
  const data = storage.data;

  if (storage.dtype === 'float64') {
    f64Buf.set((data as Float64Array).subarray(off, off + totalSize));
  } else if (storage.dtype === 'float16') {
    const f32Data = f16ToF32Input(
      data.subarray(off, off + totalSize) as TypedArray,
      storage.dtype
    ) as Float32Array;
    for (let i = 0; i < totalSize; i++) f64Buf[i] = f32Data[i]!;
  } else if (isBigIntDType(storage.dtype)) {
    const typed = data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < totalSize; i++) f64Buf[i] = Number(typed[off + i]!);
  } else {
    for (let i = 0; i < totalSize; i++) f64Buf[i] = Number(data[off + i]!);
  }

  const inputPtr = scratchCopyIn(f64Buf as unknown as TypedArray);
  const scratchPtr = scratchAlloc(axisSize * 8);

  reduce_quantile_strided_f64(inputPtr, outRegion.ptr, scratchPtr, outer, axisSize, inner, q);

  return ArrayStorage.fromWasmRegion(
    [outSize],
    'float64',
    outRegion,
    outSize,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
