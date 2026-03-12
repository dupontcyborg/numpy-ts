/**
 * WASM-accelerated least-squares solve via QR decomposition.
 *
 * Solves Ax = b for overdetermined systems (m >= n) using Householder QR.
 * Only handles single RHS (1D b). For multi-RHS, falls back to JS/SVD.
 * Returns null if WASM can't handle this case.
 */

import { lstsq_f64 } from './bins/qr.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM

/**
 * WASM-accelerated least-squares solve for 2D A and 1D b.
 * Returns x[n] or null.
 */
export function wasmLstsq(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2) return null;
  if (b.ndim !== 1) return null; // Only single RHS

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (m < n) return null; // Must be overdetermined or square
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const k = Math.min(m, n);

  // Scratch layout: a_copy[m*n] + Q[m*k] + R[k*n] + tau[k] + QtB[k] + qr_scratch[k]
  const aSize = m * n;
  const scratchF64 = aSize + m * k + k * n + k + k + k;
  const totalBytes = (aSize + n + scratchF64) * 8;

  ensureMemory(totalBytes);
  resetAllocator();

  // Copy A to WASM memory (converting to float64)
  const aData = new Float64Array(aSize);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      aData[i * n + j] = Number(a.get(i, j));
    }
  }

  // Copy b to WASM memory
  const bData = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    bData[i] = Number(b.get(i));
  }

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const xPtr = alloc(n * 8);
  const scratchPtr = alloc(scratchF64 * 8);

  lstsq_f64(aPtr, bPtr, xPtr, scratchPtr, m, n);

  const xData = copyOut(xPtr, n, Float64Array);

  return ArrayStorage.fromData(xData, [n], 'float64');
}
