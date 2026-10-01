/**
 * WASM-accelerated Householder QR decomposition.
 *
 * Computes A = Q·R for A[m×n], where Q[m×k] is orthogonal and R[k×n] is upper triangular.
 * k = min(m, n). Only supports float64 (matches JS behavior of converting all inputs to float64).
 * Returns null if WASM can't handle this case.
 */

import { isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import { qr_c128, qr_f64 } from './bins/qr.wasm';
import { wasmConfig } from './config';
import { getSharedMemory, resetScratchAllocator, scratchAlloc, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (QR is O(n³), worth it even for small)

/**
 * WASM-accelerated QR decomposition for 2D float64 matrices.
 * Returns { q: ArrayStorage, r: ArrayStorage } or null.
 */
export function wasmQr(a: ArrayStorage): { q: ArrayStorage; r: ArrayStorage } | null {
  if (a.ndim !== 2) return null;

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
      length: number,
    ) => TypedArray,
  );
  const rStorage = ArrayStorage.fromWasmRegion(
    [k, n],
    'float64',
    rRegion,
    rSize,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number,
    ) => TypedArray,
  );

  return { q: qStorage, r: rStorage };
}

/**
 * WASM-accelerated QR decomposition for 2D complex128 matrices.
 *
 * complex64 is not routed here: the kernel works in f64, so a complex64 input
 * would be promoted and come back wider than NumPy returns it.
 *
 * @param a - Complex matrix
 * @returns { q, r } with A = Q R, or null when WASM cannot take this case
 */
export function wasmQrComplex(a: ArrayStorage): { q: ArrayStorage; r: ArrayStorage } | null {
  if (a.ndim !== 2) return null;
  if (a.dtype !== 'complex128') return null;

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const k = Math.min(m, n);
  const qSlots = m * k * 2;
  const rSlots = k * n * 2;

  const qRegion = wasmMalloc(qSlots * 8);
  if (!qRegion) return null;
  const rRegion = wasmMalloc(rSlots * 8);
  if (!rRegion) {
    qRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // The kernel reduces its input in place, so it gets a working copy.
  const aSlots = m * n * 2;
  const aRegion = wasmMalloc(aSlots * 8);
  if (!aRegion) {
    qRegion.release();
    rRegion.release();
    return null;
  }
  const mem = getSharedMemory();
  const aView = new Float64Array(mem.buffer, aRegion.ptr, aSlots);
  if (a.isCContiguous) {
    if (a.isWasmBacked) {
      aView.set(new Float64Array(mem.buffer, a.wasmPtr + a.offset * 16, aSlots));
    } else {
      aView.set((a.data as Float64Array).subarray(a.offset * 2, a.offset * 2 + aSlots));
    }
  } else {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const v = a.get(i, j);
        const c = v as { re?: number; im?: number };
        aView[(i * n + j) * 2] = typeof c?.re === 'number' ? c.re : Number(v);
        aView[(i * n + j) * 2 + 1] = typeof c?.im === 'number' ? c.im : 0;
      }
    }
  }

  const tauPtr = scratchAlloc(k * 8);
  const scratchPtr = scratchAlloc(k * 2 * 8);

  qr_c128(aRegion.ptr, qRegion.ptr, rRegion.ptr, tauPtr, scratchPtr, m, n);
  aRegion.release();

  const ctor = Float64Array as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number,
  ) => TypedArray;

  return {
    q: ArrayStorage.fromWasmRegion([m, k], 'complex128', qRegion, qSlots, ctor),
    r: ArrayStorage.fromWasmRegion([k, n], 'complex128', rRegion, rSlots, ctor),
  };
}
