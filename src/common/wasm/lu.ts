/**
 * WASM-accelerated LU decomposition with partial pivoting.
 *
 * Provides lu_factor, lu_solve, and lu_inv for f64 and f32 square matrices.
 */

import {
  lu_factor_f64,
  lu_inv_f64,
  lu_solve_f64,
  lu_factor_f32,
  lu_inv_f32,
  lu_solve_f32,
} from './bins/lu.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  getSharedMemory,
} from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

/**
 * WASM LU factorization. Returns { lu, piv, sign } or null.
 * lu: n×n f64 ArrayStorage with L (below diag) and U (on+above diag).
 * piv: Int32Array of pivot indices.
 * sign: +1 or -1 (permutation sign for determinant).
 */
export function wasmLuFactor(
  a: ArrayStorage
): { lu: ArrayStorage; piv: Int32Array; sign: number } | null {
  if (!a.isCContiguous) return null;
  if (a.ndim !== 2) return null;
  const [m, n] = a.shape;
  if (m !== n || m! < 2) return null;

  const dtype = a.dtype;
  const size = m!;
  const isF32 = dtype === 'float32';
  if (dtype !== 'float64' && dtype !== 'float32') return null;

  const bpe = isF32 ? 4 : 8;
  const matBytes = size * size * bpe;
  const pivBytes = size * 4; // i32

  // Allocate output for LU matrix (will be modified in-place)
  const luRegion = wasmMalloc(matBytes);
  if (!luRegion) return null;

  const pivRegion = wasmMalloc(pivBytes);
  if (!pivRegion) {
    luRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Copy input into LU region
  const mem = getSharedMemory();
  const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size * size, bpe);
  new Uint8Array(mem.buffer, luRegion.ptr, matBytes).set(
    new Uint8Array(mem.buffer, inPtr, matBytes)
  );

  // Factor in-place
  const sign = isF32
    ? lu_factor_f32(luRegion.ptr, pivRegion.ptr, size)
    : lu_factor_f64(luRegion.ptr, pivRegion.ptr, size);

  const Ctor = isF32
    ? (Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray)
    : (Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray);

  const lu = ArrayStorage.fromWasmRegion([size, size], dtype, luRegion, size * size, Ctor);
  const piv = new Int32Array(mem.buffer, pivRegion.ptr, size).slice(); // copy out
  pivRegion.release();

  return { lu, piv, sign };
}

/**
 * WASM LU inverse. Takes pre-computed LU + piv, returns n×n inverse.
 */
export function wasmLuInv(lu: ArrayStorage, piv: Int32Array, dtype: DType): ArrayStorage | null {
  const size = lu.shape[0]!;
  const isF32 = dtype === 'float32';
  const bpe = isF32 ? 4 : 8;
  const matBytes = size * size * bpe;

  const outRegion = wasmMalloc(matBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Copy piv to scratch
  const pivPtr = scratchAlloc(size * 4);
  const mem = getSharedMemory();
  new Int32Array(mem.buffer, pivPtr, size).set(piv);

  const luPtr = resolveInputPtr(lu.data, lu.isWasmBacked, lu.wasmPtr, lu.offset, size * size, bpe);

  if (isF32) {
    lu_inv_f32(luPtr, pivPtr, outRegion.ptr, size);
  } else {
    lu_inv_f64(luPtr, pivPtr, outRegion.ptr, size);
  }

  const Ctor = isF32
    ? (Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray)
    : (Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray);

  return ArrayStorage.fromWasmRegion([size, size], dtype, outRegion, size * size, Ctor);
}

/**
 * WASM LU solve. Solves LU @ x = b for a single RHS vector.
 */
export function wasmLuSolve(
  lu: ArrayStorage,
  piv: Int32Array,
  b: ArrayStorage,
  dtype: DType
): ArrayStorage | null {
  const size = lu.shape[0]!;
  const isF32 = dtype === 'float32';
  const bpe = isF32 ? 4 : 8;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const pivPtr = scratchAlloc(size * 4);
  const mem = getSharedMemory();
  new Int32Array(mem.buffer, pivPtr, size).set(piv);

  const luPtr = resolveInputPtr(lu.data, lu.isWasmBacked, lu.wasmPtr, lu.offset, size * size, bpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);

  if (isF32) {
    lu_solve_f32(luPtr, pivPtr, bPtr, outRegion.ptr, size);
  } else {
    lu_solve_f64(luPtr, pivPtr, bPtr, outRegion.ptr, size);
  }

  const Ctor = isF32
    ? (Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray)
    : (Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray);

  return ArrayStorage.fromWasmRegion([size], dtype, outRegion, size, Ctor);
}
