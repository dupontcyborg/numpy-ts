/**
 * WASM-accelerated Singular Value Decomposition.
 *
 * Computes A[m×n] = U[m×m] · diag(S) · Vt[n×n] via Jacobi eigendecomposition of A^T·A.
 * Only supports float64 (matches JS behavior of converting all inputs to float64).
 * Returns null if WASM can't handle this case.
 */

import { svd_f64 } from './bins/svd.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (SVD is O(n³), worth it even for small)

/**
 * WASM-accelerated full SVD for 2D float64 matrices.
 * Returns { u: ArrayStorage, s: ArrayStorage, vt: ArrayStorage } or null.
 */
export function wasmSvd(
  a: ArrayStorage
): { u: ArrayStorage; s: ArrayStorage; vt: ArrayStorage } | null {
  if (a.ndim !== 2) return null;

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const k = Math.min(m, n);

  // Memory layout: a[m*n] + u[m*m] + s[k] + vt[n*n] + work[m*n + n*n]
  const aSize = m * n;
  const uSize = m * m;
  const sSize = k;
  const vtSize = n * n;
  const workSize = m * n + n * n;
  const totalF64 = aSize + uSize + sSize + vtSize + workSize;
  const totalBytes = totalF64 * 8; // f64 = 8 bytes

  ensureMemory(totalBytes);
  resetAllocator();

  // Copy input matrix to WASM memory (converting to float64)
  const aData = new Float64Array(aSize);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      aData[i * n + j] = Number(a.get(i, j));
    }
  }

  const aPtr = copyIn(aData);
  const uPtr = alloc(uSize * 8);
  const sPtr = alloc(sSize * 8);
  const vtPtr = alloc(vtSize * 8);
  const workPtr = alloc(workSize * 8);

  svd_f64(aPtr, uPtr, sPtr, vtPtr, workPtr, m, n);

  const uData = copyOut(uPtr, uSize, Float64Array);
  const sData = copyOut(sPtr, sSize, Float64Array);
  const vtData = copyOut(vtPtr, vtSize, Float64Array);

  const uStorage = ArrayStorage.fromData(uData, [m, m], 'float64');
  const sStorage = ArrayStorage.fromData(sData, [k], 'float64');
  const vtStorage = ArrayStorage.fromData(vtData, [n, n], 'float64');

  return { u: uStorage, s: sStorage, vt: vtStorage };
}
