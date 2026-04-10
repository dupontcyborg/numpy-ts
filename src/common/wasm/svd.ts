/**
 * WASM-accelerated Singular Value Decomposition.
 *
 * Computes A[m×n] = U[m×m] · diag(S) · Vt[n×n] via Jacobi eigendecomposition of A^T·A.
 * Only supports float64 (matches JS behavior of converting all inputs to float64).
 * Returns null if WASM can't handle this case.
 */

import { svd_f64, svd_values_gk_f64 } from './bins/svd.wasm';
import { wasmMalloc, resetScratchAllocator, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type TypedArray } from '../dtype';

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

  // TODO: support complex in WASM
  if (isComplexDType(a.dtype)) return null;

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const k = Math.min(m, n);

  const uSize = m * m;
  const sSize = k;
  const vtSize = n * n;

  // Allocate persistent output for U, S, Vt
  const uRegion = wasmMalloc(uSize * 8);
  if (!uRegion) return null;
  const sRegion = wasmMalloc(sSize * 8);
  if (!sRegion) {
    uRegion.release();
    return null;
  }
  const vtRegion = wasmMalloc(vtSize * 8);
  if (!vtRegion) {
    uRegion.release();
    sRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // SVD modifies input during Householder bidiagonalization — allocate working copy on heap
  const aSize = m * n;
  const aRegion = wasmMalloc(aSize * 8);
  if (!aRegion) {
    uRegion.release();
    sRegion.release();
    vtRegion.release();
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
  const workSize = m * n + n * n;
  const workRegion = wasmMalloc(workSize * 8);
  if (!workRegion) {
    aRegion.release();
    uRegion.release();
    sRegion.release();
    vtRegion.release();
    return null;
  }

  svd_f64(aRegion.ptr, uRegion.ptr, sRegion.ptr, vtRegion.ptr, workRegion.ptr, m, n);
  aRegion.release();
  workRegion.release();

  const F64Ctor = Float64Array as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number
  ) => TypedArray;

  const uStorage = ArrayStorage.fromWasmRegion([m, m], 'float64', uRegion, uSize, F64Ctor);
  const sStorage = ArrayStorage.fromWasmRegion([k], 'float64', sRegion, sSize, F64Ctor);
  const vtStorage = ArrayStorage.fromWasmRegion([n, n], 'float64', vtRegion, vtSize, F64Ctor);

  return { u: uStorage, s: sStorage, vt: vtStorage };
}

/**
 * WASM-accelerated singular values only (no U, V) via Golub-Kahan.
 * Much faster than full SVD for svdvals/cond/matrix_rank.
 * Returns ArrayStorage with singular values, or null.
 */
export function wasmSvdValues(a: ArrayStorage): ArrayStorage | null {
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

  const sRegion = wasmMalloc(k * 8);
  if (!sRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // GK-SVD modifies input — allocate working copy on heap
  const aSize = m * n;
  const aRegion = wasmMalloc(aSize * 8);
  if (!aRegion) {
    sRegion.release();
    return null;
  }
  const mem = getSharedMemory();
  if (a.isCContiguous && a.dtype === 'float64') {
    const aView = new Float64Array(mem.buffer, aRegion.ptr, aSize);
    if (a.isWasmBacked) {
      aView.set(new Float64Array(mem.buffer, a.wasmPtr + a.offset * 8, aSize));
    } else {
      aView.set((a.data as Float64Array).subarray(a.offset, a.offset + aSize));
    }
  } else {
    const aView = new Float64Array(mem.buffer, aRegion.ptr, aSize);
    for (let i = 0; i < aSize; i++) aView[i] = Number(a.iget(i));
  }
  // Workspace for GK iteration — can be large, allocate on heap
  const workSize = m * n + 4 * k;
  const workRegion = wasmMalloc(workSize * 8);
  if (!workRegion) {
    aRegion.release();
    sRegion.release();
    return null;
  }

  svd_values_gk_f64(aRegion.ptr, sRegion.ptr, workRegion.ptr, m, n);
  aRegion.release();
  workRegion.release();

  const F64Ctor = Float64Array as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number
  ) => TypedArray;

  return ArrayStorage.fromWasmRegion([k], 'float64', sRegion, k, F64Ctor);
}
