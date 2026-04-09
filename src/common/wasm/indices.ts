/**
 * WASM-accelerated np.indices() for 2D and 3D grids.
 * Supports int32, int64, and float64 output dtypes.
 */

import {
  indices_2d_i32,
  indices_3d_i32,
  indices_2d_i64,
  indices_3d_i64,
  indices_2d_f64,
  indices_3d_f64,
} from './bins/indices.wasm';
import { wasmMalloc, resetScratchAllocator } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type IndicesKernel2D = (out: number, d0: number, d1: number) => void;
type IndicesKernel3D = (out: number, d0: number, d1: number, d2: number) => void;

const kernels2D: Partial<Record<DType, IndicesKernel2D>> = {
  int32: indices_2d_i32,
  int64: indices_2d_i64,
  float64: indices_2d_f64,
};

const kernels3D: Partial<Record<DType, IndicesKernel3D>> = {
  int32: indices_3d_i32,
  int64: indices_3d_i64,
  float64: indices_3d_f64,
};

const bpeMap: Partial<Record<DType, number>> = {
  int32: 4,
  int64: 8,
  float64: 8,
};

const ctorMap: Partial<Record<DType, new (buf: ArrayBuffer, off: number, len: number) => TypedArray>> = {
  int32: Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int64: BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  float64: Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
};

/**
 * WASM-accelerated indices for 2D/3D grids.
 * Returns null if WASM can't handle this case.
 */
export function wasmIndices(dimensions: number[], dtype: string): ArrayStorage | null {
  const k2d = kernels2D[dtype as DType];
  const k3d = kernels3D[dtype as DType];
  const bpe = bpeMap[dtype as DType];
  const Ctor = ctorMap[dtype as DType];
  if ((!k2d && !k3d) || !bpe || !Ctor) return null;

  const ndim = dimensions.length;
  if (ndim < 2 || ndim > 3) return null;

  const gridSize = dimensions.reduce((a, b) => a * b, 1);
  if (gridSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const totalSize = ndim * gridSize;
  const outRegion = wasmMalloc(totalSize * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (ndim === 2) {
    k2d!(outRegion.ptr, dimensions[0]!, dimensions[1]!);
  } else {
    k3d!(outRegion.ptr, dimensions[0]!, dimensions[1]!, dimensions[2]!);
  }

  return ArrayStorage.fromWasmRegion(
    [ndim, ...dimensions],
    dtype as DType,
    outRegion,
    totalSize,
    Ctor
  );
}
