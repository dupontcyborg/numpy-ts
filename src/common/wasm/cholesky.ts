/**
 * WASM-accelerated Cholesky decomposition.
 *
 * Computes A = L·L^T for symmetric positive-definite A[n×n], or A = L·L^H for
 * Hermitian positive-definite complex A. Supports float64, float32,
 * complex128 and complex64. Returns null if WASM can't handle this case.
 */

import { isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import { cholesky_c64, cholesky_c128, cholesky_f32, cholesky_f64 } from './bins/cholesky.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr, scratchCopyIn, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (Cholesky is O(n³), worth it even for small)

/**
 * WASM-accelerated Cholesky decomposition for 2D float64 matrices.
 * Returns ArrayStorage (lower triangular L) or null.
 * Throws if the matrix is not positive-definite.
 */
export function wasmCholesky(a: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2) return null;
  if (isComplexDType(a.dtype)) return null;

  const n = a.shape[0]!;
  if (n !== a.shape[1]!) return null; // Must be square

  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const matSize = n * n;
  const outBytes = matSize * 8;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Resolve input — zero-copy if already WASM-backed float64 contiguous
  let aPtr: number;
  if (a.isCContiguous && a.dtype === 'float64') {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, matSize, 8);
  } else {
    const aData = new Float64Array(matSize);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aData[i * n + j] = Number(a.get(i, j));
      }
    }
    aPtr = scratchCopyIn(aData as unknown as TypedArray);
  }

  const rc = cholesky_f64(aPtr, outRegion.ptr, n);

  if (rc !== 0) {
    outRegion.release();
    throw new Error('cholesky: matrix is not positive definite');
  }

  return ArrayStorage.fromWasmRegion(
    [n, n],
    'float64',
    outRegion,
    matSize,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );
}

/**
 * WASM-accelerated Cholesky decomposition for 2D complex matrices.
 *
 * Reads the lower triangle, as NumPy does, and returns L with A = L·L^H.
 * Returns ArrayStorage or null; throws if the matrix is not positive-definite.
 *
 * @param a - Hermitian positive-definite complex matrix
 * @returns Lower triangular L, or null when WASM cannot take this case
 */
export function wasmCholeskyComplex(a: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2) return null;
  const isC64 = a.dtype === 'complex64';
  if (a.dtype !== 'complex128' && !isC64) return null;

  const n = a.shape[0]!;
  if (n !== a.shape[1]!) return null;
  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const matSize = n * n;
  const slots = matSize * 2; // interleaved [re, im]
  const bytesPerSlot = isC64 ? 4 : 8;
  const outRegion = wasmMalloc(slots * bytesPerSlot);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  let aPtr: number;
  if (a.isCContiguous) {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, slots, bytesPerSlot);
  } else {
    const aData = isC64 ? new Float32Array(slots) : new Float64Array(slots);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = a.get(i, j);
        const c = v as { re?: number; im?: number };
        aData[(i * n + j) * 2] = typeof c?.re === 'number' ? c.re : Number(v);
        aData[(i * n + j) * 2 + 1] = typeof c?.im === 'number' ? c.im : 0;
      }
    }
    aPtr = scratchCopyIn(aData as unknown as TypedArray);
  }

  const rc = isC64 ? cholesky_c64(aPtr, outRegion.ptr, n) : cholesky_c128(aPtr, outRegion.ptr, n);

  if (rc !== 0) {
    outRegion.release();
    throw new Error('cholesky: matrix is not positive definite');
  }

  return ArrayStorage.fromWasmRegion(
    [n, n],
    isC64 ? 'complex64' : 'complex128',
    outRegion,
    slots,
    (isC64 ? Float32Array : Float64Array) as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );
}

/**
 * WASM-accelerated Cholesky decomposition for 2D float32 matrices.
 * Returns ArrayStorage (lower triangular L) or null.
 * Throws if the matrix is not positive-definite.
 */
export function wasmCholeskyF32(a: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2) return null;
  if (isComplexDType(a.dtype)) return null;

  const n = a.shape[0]!;
  if (n !== a.shape[1]!) return null;

  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const matSize = n * n;
  const outBytes = matSize * 4;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  let aPtr: number;
  if (a.isCContiguous && a.dtype === 'float32') {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, matSize, 4);
  } else {
    const aData = new Float32Array(matSize);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aData[i * n + j] = Number(a.get(i, j));
      }
    }
    aPtr = scratchCopyIn(aData as unknown as TypedArray);
  }

  const rc = cholesky_f32(aPtr, outRegion.ptr, n);

  if (rc !== 0) {
    outRegion.release();
    throw new Error('cholesky: matrix is not positive definite');
  }

  return ArrayStorage.fromWasmRegion(
    [n, n],
    'float32',
    outRegion,
    matSize,
    Float32Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );
}
