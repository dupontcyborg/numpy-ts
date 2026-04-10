/**
 * WASM-accelerated Cholesky decomposition.
 *
 * Computes A = L·L^T for symmetric positive-definite A[n×n].
 * Supports float64 and float32. Returns null if WASM can't handle this case.
 */

import { cholesky_f64, cholesky_f32 } from './bins/cholesky.wasm';
import { wasmMalloc, resetScratchAllocator, scratchCopyIn, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

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
      length: number
    ) => TypedArray
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
      length: number
    ) => TypedArray
  );
}
