/**
 * WASM-accelerated LU decomposition with partial pivoting.
 *
 * Provides lu_factor, lu_solve, and lu_inv for f64, f32 and complex square
 * matrices. The complex kernels are c128 only; complex64 input is widened on
 * the way in, which is what the JS path does too, so the result dtype is
 * unchanged by the routing.
 */

import type { DType, TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  lu_factor_c64,
  lu_factor_c128,
  lu_factor_f32,
  lu_factor_f64,
  lu_inv_c64,
  lu_inv_c128,
  lu_inv_f32,
  lu_inv_f64,
  lu_solve_c64,
  lu_solve_c128,
  lu_solve_f32,
  lu_solve_f64,
} from './bins/lu.wasm';
import { wasmConfig } from './config';
import {
  getSharedMemory,
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  scratchCopyIn,
  wasmMalloc,
} from './runtime';

/**
 * WASM LU factorization. Returns { lu, piv, sign } or null.
 * lu: n×n f64 ArrayStorage with L (below diag) and U (on+above diag).
 * piv: Int32Array of pivot indices.
 * sign: +1 or -1 (permutation sign for determinant).
 */
export function wasmLuFactor(
  a: ArrayStorage,
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
    new Uint8Array(mem.buffer, inPtr, matBytes),
  );

  // Factor in-place
  const sign = isF32
    ? lu_factor_f32(luRegion.ptr, pivRegion.ptr, size)
    : lu_factor_f64(luRegion.ptr, pivRegion.ptr, size);

  const Ctor = isF32
    ? (Float32Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray)
    : (Float64Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray);

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
    ? (Float32Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray)
    : (Float64Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray);

  return ArrayStorage.fromWasmRegion([size, size], dtype, outRegion, size * size, Ctor);
}

/**
 * WASM LU solve. Solves LU @ x = b for a single RHS vector.
 */
export function wasmLuSolve(
  lu: ArrayStorage,
  piv: Int32Array,
  b: ArrayStorage,
  dtype: DType,
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
    ? (Float32Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray)
    : (Float64Array as unknown as new (
        buf: ArrayBuffer,
        off: number,
        len: number,
      ) => TypedArray);

  return ArrayStorage.fromWasmRegion([size], dtype, outRegion, size, Ctor);
}

/**
 * WASM complex LU factorization. `outDtype` picks the precision the kernel runs
 * in: complex128 widens a complex64 input on the way in, which is what the JS
 * path does and what det and slogdet want, while complex64 keeps it single for
 * callers that have to hand back a complex64 result.
 *
 * @param a - Square complex matrix
 * @param outDtype - Precision to factor in, and the dtype of the returned lu
 * @returns { lu, piv, sign } or null to fall back to JS
 */
export function wasmLuFactorComplex(
  a: ArrayStorage,
  outDtype: 'complex64' | 'complex128' = 'complex128',
): { lu: ArrayStorage; piv: Int32Array; sign: number } | null {
  if (a.ndim !== 2) return null;
  if (a.dtype !== 'complex128' && a.dtype !== 'complex64') return null;
  if (outDtype === 'complex64' && a.dtype !== 'complex64') return null;
  const [m, n] = a.shape;
  if (m !== n || m! < 2) return null;

  // No size floor: the kernel beats the JS loop from 2x2 up. The comparison is
  // still written against thresholdMultiplier so that setting it to Infinity
  // turns this path off, which is how FORCE_BACKEND=js disables WASM.
  if (m! < 2 * wasmConfig.thresholdMultiplier) return null;

  const size = m!;
  const slots = size * size * 2;
  const isC64 = outDtype === 'complex64';
  const bpe = isC64 ? 4 : 8;
  const Arr = isC64 ? Float32Array : Float64Array;

  const luRegion = wasmMalloc(slots * bpe);
  if (!luRegion) return null;

  const pivRegion = wasmMalloc(size * 4);
  if (!pivRegion) {
    luRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const mem = getSharedMemory();
  const luView = new Arr(mem.buffer, luRegion.ptr, slots);
  if (a.dtype === outDtype && a.isCContiguous) {
    if (a.isWasmBacked) {
      luView.set(new Arr(mem.buffer, a.wasmPtr + a.offset * 2 * bpe, slots));
    } else {
      luView.set((a.data as typeof luView).subarray(a.offset * 2, a.offset * 2 + slots));
    }
  } else {
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const v = a.get(i, j) as { re?: number; im?: number };
        luView[(i * size + j) * 2] = typeof v?.re === 'number' ? v.re : Number(v);
        luView[(i * size + j) * 2 + 1] = typeof v?.im === 'number' ? v.im : 0;
      }
    }
  }

  const sign = isC64
    ? lu_factor_c64(luRegion.ptr, pivRegion.ptr, size)
    : lu_factor_c128(luRegion.ptr, pivRegion.ptr, size);

  const ctor = Arr as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray;
  const lu = ArrayStorage.fromWasmRegion([size, size], outDtype, luRegion, slots, ctor);
  const piv = new Int32Array(mem.buffer, pivRegion.ptr, size).slice();
  pivRegion.release();

  return { lu, piv, sign };
}

/**
 * WASM complex LU inverse, in the precision of the LU factors it is given.
 *
 * @param lu - Packed complex LU factors from wasmLuFactorComplex
 * @param piv - Pivot permutation from the factorization
 * @returns Inverse with the same dtype as lu, or null to fall back to JS
 */
export function wasmLuInvComplex(lu: ArrayStorage, piv: Int32Array): ArrayStorage | null {
  const size = lu.shape[0]!;
  const slots = size * size * 2;
  const isC64 = lu.dtype === 'complex64';
  const bpe = isC64 ? 4 : 8;

  const outRegion = wasmMalloc(slots * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const pivPtr = scratchAlloc(size * 4);
  const mem = getSharedMemory();
  new Int32Array(mem.buffer, pivPtr, size).set(piv);

  const luPtr = resolveInputPtr(lu.data, lu.isWasmBacked, lu.wasmPtr, lu.offset * 2, slots, bpe);
  if (isC64) {
    lu_inv_c64(luPtr, pivPtr, outRegion.ptr, size);
  } else {
    lu_inv_c128(luPtr, pivPtr, outRegion.ptr, size);
  }

  const ctor = (isC64 ? Float32Array : Float64Array) as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number,
  ) => TypedArray;
  return ArrayStorage.fromWasmRegion([size, size], lu.dtype, outRegion, slots, ctor);
}

/**
 * WASM complex LU solve for a single RHS vector, in the precision of the LU
 * factors. The kernel applies the pivot permutation itself, so `rhs` goes in
 * unpermuted.
 *
 * @param lu - Packed complex LU factors from wasmLuFactorComplex
 * @param piv - Pivot permutation from the factorization
 * @param rhs - Interleaved [re, im] right-hand side, 2·n values
 * @returns Solution with the same dtype as lu, or null to fall back to JS
 */
export function wasmLuSolveComplex(
  lu: ArrayStorage,
  piv: Int32Array,
  rhs: Float64Array | Float32Array,
): ArrayStorage | null {
  const size = lu.shape[0]!;
  const slots = size * size * 2;
  const isC64 = lu.dtype === 'complex64';
  const bpe = isC64 ? 4 : 8;

  const outRegion = wasmMalloc(size * 2 * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const pivPtr = scratchAlloc(size * 4);
  const mem = getSharedMemory();
  new Int32Array(mem.buffer, pivPtr, size).set(piv);

  const bPtr = scratchCopyIn(rhs as unknown as TypedArray);
  const luPtr = resolveInputPtr(lu.data, lu.isWasmBacked, lu.wasmPtr, lu.offset * 2, slots, bpe);
  if (isC64) {
    lu_solve_c64(luPtr, pivPtr, bPtr, outRegion.ptr, size);
  } else {
    lu_solve_c128(luPtr, pivPtr, bPtr, outRegion.ptr, size);
  }

  const ctor = (isC64 ? Float32Array : Float64Array) as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number,
  ) => TypedArray;
  return ArrayStorage.fromWasmRegion([size], lu.dtype, outRegion, size * 2, ctor);
}
