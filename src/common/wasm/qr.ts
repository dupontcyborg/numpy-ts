/**
 * WASM-accelerated Householder QR decomposition.
 *
 * Computes A = Q·R for A[m×n], where Q[m×k] is orthogonal and R[k×n] is upper triangular.
 * k = min(m, n). Only supports float64 (matches JS behavior of converting all inputs to float64).
 * Returns null if WASM can't handle this case.
 */

import { qr_f64 } from './bins/qr.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (QR is O(n³), worth it even for small)

/**
 * WASM-accelerated QR decomposition for 2D float64 matrices.
 * Returns { q: ArrayStorage, r: ArrayStorage } or null.
 */
export function wasmQr(a: ArrayStorage): { q: ArrayStorage; r: ArrayStorage } | null {
  if (a.ndim !== 2) return null;

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const k = Math.min(m, n);

  // Memory layout: a_copy[m*n] + q[m*k] + r[k*n] + tau[k] + scratch[k]
  const aSize = m * n;
  const qSize = m * k;
  const rSize = k * n;
  const tauSize = k;
  const scratchSize = k;
  const totalF64 = aSize + qSize + rSize + tauSize + scratchSize;
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
  const qPtr = alloc(qSize * 8);
  const rPtr = alloc(rSize * 8);
  const tauPtr = alloc(tauSize * 8);
  const scratchPtr = alloc(scratchSize * 8);

  qr_f64(aPtr, qPtr, rPtr, tauPtr, scratchPtr, m, n);

  const qData = copyOut(qPtr, qSize, Float64Array);
  const rData = copyOut(rPtr, rSize, Float64Array);

  const qStorage = ArrayStorage.fromData(qData, [m, k], 'float64');
  const rStorage = ArrayStorage.fromData(rData, [k, n], 'float64');

  return { q: qStorage, r: rStorage };
}
