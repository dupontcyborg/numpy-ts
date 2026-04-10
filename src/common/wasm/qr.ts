/**
 * WASM-accelerated Householder QR decomposition.
 *
 * Computes A = Q·R for A[m×n], where Q[m×k] is orthogonal and R[k×n] is upper triangular.
 * k = min(m, n). Only supports float64 (matches JS behavior of converting all inputs to float64).
 * Returns null if WASM can't handle this case.
 */

import { qr_f64 } from './bins/qr.wasm';
import { wasmMalloc, resetScratchAllocator, scratchAlloc, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (QR is O(n³), worth it even for small)

/**
 * WASM-accelerated QR decomposition for 2D float64 matrices.
 * Returns { q: ArrayStorage, r: ArrayStorage } or null.
 */
export function wasmQr(a: ArrayStorage): { q: ArrayStorage; r: ArrayStorage } | null {
  if (a.ndim !== 2) return null;

  // TODO: Add complex128/complex64 support to WASM
  if (isComplexDType(a.dtype)) return null;

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const k = Math.min(m, n);

  const qSize = m * k;
  const rSize = k * n;

  // Allocate persistent output for Q and R
  const qRegion = wasmMalloc(qSize * 8);
  if (!qRegion) return null;
  const rRegion = wasmMalloc(rSize * 8);
  if (!rRegion) {
    qRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // QR modifies input during Householder reflections — allocate working copy on heap
  const aSize = m * n;
  const aRegion = wasmMalloc(aSize * 8);
  if (!aRegion) {
    qRegion.release();
    rRegion.release();
    return null;
  }
  const mem = getSharedMemory();
  if (a.dtype === 'float64' && a.isCContiguous) {
    const aView = new Float64Array(mem.buffer, aRegion.ptr, aSize);
    if (a.isWasmBacked) {
      aView.set(new Float64Array(mem.buffer, a.wasmPtr + a.offset * 8, aSize));
    } else {
      aView.set((a.data as Float64Array).subarray(a.offset, a.offset + aSize));
    }
  } else {
    const aView = new Float64Array(mem.buffer, aRegion.ptr, aSize);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        aView[i * n + j] = Number(a.get(i, j));
      }
    }
  }
  const tauPtr = scratchAlloc(k * 8);
  const scratchPtr = scratchAlloc(k * 8);

  qr_f64(aRegion.ptr, qRegion.ptr, rRegion.ptr, tauPtr, scratchPtr, m, n);
  aRegion.release();

  const qStorage = ArrayStorage.fromWasmRegion(
    [m, k],
    'float64',
    qRegion,
    qSize,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
  const rStorage = ArrayStorage.fromWasmRegion(
    [k, n],
    'float64',
    rRegion,
    rSize,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );

  return { q: qStorage, r: rStorage };
}
