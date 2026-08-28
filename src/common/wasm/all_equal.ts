/**
 * WASM-accelerated whole-array equality, backing array_equal / array_equiv.
 *
 * The JS loop this fronts is already monomorphic and fine per element — the
 * issue is that it costs the same per element for every dtype while NumPy's
 * cost scales with bytes. Comparing a v128 at a time puts us on the same
 * footing: 16 lanes for int8 down to 2 for int64/float64.
 *
 * Returns null when WASM can't take the case, and the caller falls through.
 */

import type { DType, TypedArray } from '../dtype';
import type { ArrayStorage } from '../storage';
import {
  all_equal_f32,
  all_equal_f64,
  all_equal_i8,
  all_equal_i16,
  all_equal_i32,
  all_equal_i64,
  all_equal_nan_f32,
  all_equal_nan_f64,
  all_equal_u8,
  all_equal_u16,
  all_equal_u32,
  all_equal_u64,
} from './bins/all_equal.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr } from './runtime';

/**
 * Below this the kernel call costs more than the loop it replaces. At default
 * benchmark sizes the JS path is only 2-7x off NumPy on 7-10us of work, where
 * call overhead would eat the win; the gap opens up at scale.
 */
const BASE_THRESHOLD = 4096;

type EqFn = (aPtr: number, bPtr: number, N: number) => number;

/** float16 is deliberately absent: we already run it at 0.7-0.8x, faster than NumPy. */
const kernels: Partial<Record<DType, EqFn>> = {
  float64: all_equal_f64,
  float32: all_equal_f32,
  int64: all_equal_i64,
  uint64: all_equal_u64,
  int32: all_equal_i32,
  uint32: all_equal_u32,
  int16: all_equal_i16,
  uint16: all_equal_u16,
  int8: all_equal_i8,
  uint8: all_equal_u8,
  bool: all_equal_u8,
};

/** equal_nan only changes float behaviour — integers cannot hold NaN. */
const nanKernels: Partial<Record<DType, EqFn>> = {
  float64: all_equal_nan_f64,
  float32: all_equal_nan_f32,
};

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
  bool: 1,
};

/**
 * True/false when the kernel handled it, null when it could not.
 *
 * Callers must have already established that the two arrays share a dtype and
 * a size; this only adds the contiguity, threshold and dtype-coverage checks.
 */
export function wasmAllEqual(a: ArrayStorage, b: ArrayStorage, equalNan: boolean): boolean | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  if (a.dtype !== b.dtype) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = equalNan ? (nanKernels[dtype] ?? kernels[dtype]) : kernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(
    a.data as TypedArray,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset,
    size,
    bpe,
  );
  const bPtr = resolveInputPtr(
    b.data as TypedArray,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset,
    size,
    bpe,
  );

  return kernel(aPtr, bPtr, size) === 1;
}
