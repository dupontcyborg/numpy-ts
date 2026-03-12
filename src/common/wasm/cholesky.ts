/**
 * WASM-accelerated Cholesky decomposition.
 *
 * Computes A = L·L^T for symmetric positive-definite A[n×n].
 * Supports float64 and float32. Returns null if WASM can't handle this case.
 */

import { cholesky_f64, cholesky_f32 } from './bins/cholesky.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (Cholesky is O(n³), worth it even for small)

/**
 * WASM-accelerated Cholesky decomposition for 2D float64 matrices.
 * Returns ArrayStorage (lower triangular L) or null.
 * Throws if the matrix is not positive-definite.
 */
export function wasmCholesky(a: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2) return null;

  const n = a.shape[0]!;
  if (n !== a.shape[1]!) return null; // Must be square

  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Memory layout: a[n*n] + out[n*n]
  const matSize = n * n;
  const totalF64 = matSize * 2;
  const totalBytes = totalF64 * 8; // f64 = 8 bytes

  ensureMemory(totalBytes);
  resetAllocator();

  // Copy input matrix to WASM memory (converting to float64)
  const aData = new Float64Array(matSize);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aData[i * n + j] = Number(a.get(i, j));
    }
  }

  const aPtr = copyIn(aData);
  const outPtr = alloc(matSize * 8);

  const rc = cholesky_f64(aPtr, outPtr, n);

  if (rc !== 0) {
    throw new Error('cholesky: matrix is not positive definite');
  }

  const outData = copyOut(outPtr, matSize, Float64Array);
  return ArrayStorage.fromData(outData, [n, n], 'float64');
}

/**
 * WASM-accelerated Cholesky decomposition for 2D float32 matrices.
 * Returns ArrayStorage (lower triangular L) or null.
 * Throws if the matrix is not positive-definite.
 */
export function wasmCholeskyF32(a: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2) return null;

  const n = a.shape[0]!;
  if (n !== a.shape[1]!) return null;

  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const matSize = n * n;
  const totalF32 = matSize * 2;
  const totalBytes = totalF32 * 4; // f32 = 4 bytes

  ensureMemory(totalBytes);
  resetAllocator();

  const aData = new Float32Array(matSize);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aData[i * n + j] = Number(a.get(i, j));
    }
  }

  const aPtr = copyIn(aData);
  const outPtr = alloc(matSize * 4);

  const rc = cholesky_f32(aPtr, outPtr, n);

  if (rc !== 0) {
    throw new Error('cholesky: matrix is not positive definite');
  }

  const outData = copyOut(outPtr, matSize, Float32Array);
  return ArrayStorage.fromData(outData, [n, n], 'float32');
}
