/**
 * WASM-accelerated Singular Value Decomposition.
 *
 * Computes A[m×n] = U[m×m] · diag(S) · Vt[n×n] via Jacobi eigendecomposition of A^T·A.
 * Only supports float64 (matches JS behavior of converting all inputs to float64).
 * Returns null if WASM can't handle this case.
 */

import { isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import { svd_c64, svd_c128, svd_f32, svd_f64, svd_values_gk_f64 } from './bins/svd.wasm';
import { wasmConfig } from './config';
import { getSharedMemory, resetScratchAllocator, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 4; // Minimum matrix dimension for WASM (SVD is O(n³), worth it even for small)

/**
 * WASM-accelerated full SVD for 2D float64 matrices.
 * Returns { u: ArrayStorage, s: ArrayStorage, vt: ArrayStorage } or null.
 */
export function wasmSvd(
  a: ArrayStorage,
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
  const useF32 = a.dtype === 'float32' || a.dtype === 'float16';
  const bytesPerElem = useF32 ? 4 : 8;

  const uSize = m * m;
  const sSize = k;
  const vtSize = n * n;

  // Allocate persistent output for U, S, Vt
  const uRegion = wasmMalloc(uSize * bytesPerElem);
  if (!uRegion) return null;
  const sRegion = wasmMalloc(sSize * bytesPerElem);
  if (!sRegion) {
    uRegion.release();
    return null;
  }
  const vtRegion = wasmMalloc(vtSize * bytesPerElem);
  if (!vtRegion) {
    uRegion.release();
    sRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // SVD modifies input during Householder bidiagonalization — allocate working copy on heap
  const aSize = m * n;
  const aRegion = wasmMalloc(aSize * bytesPerElem);
  if (!aRegion) {
    uRegion.release();
    sRegion.release();
    vtRegion.release();
    return null;
  }
  const mem = getSharedMemory();
  if (useF32) {
    const aView = new Float32Array(mem.buffer, aRegion.ptr, aSize);
    if (a.dtype === 'float32' && a.isCContiguous) {
      if (a.isWasmBacked) {
        aView.set(new Float32Array(mem.buffer, a.wasmPtr + a.offset * 4, aSize));
      } else {
        aView.set((a.data as Float32Array).subarray(a.offset, a.offset + aSize));
      }
    } else {
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          aView[i * n + j] = Number(a.get(i, j));
        }
      }
    }
  } else if (a.dtype === 'float64' && a.isCContiguous) {
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
  const workRegion = wasmMalloc(workSize * bytesPerElem);
  if (!workRegion) {
    aRegion.release();
    uRegion.release();
    sRegion.release();
    vtRegion.release();
    return null;
  }

  if (useF32) {
    svd_f32(aRegion.ptr, uRegion.ptr, sRegion.ptr, vtRegion.ptr, workRegion.ptr, m, n);
  } else {
    svd_f64(aRegion.ptr, uRegion.ptr, sRegion.ptr, vtRegion.ptr, workRegion.ptr, m, n);
  }
  aRegion.release();
  workRegion.release();

  const outDtype = useF32 ? 'float32' : 'float64';
  const OutCtor = (useF32 ? Float32Array : Float64Array) as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number,
  ) => TypedArray;

  const uStorage = ArrayStorage.fromWasmRegion([m, m], outDtype, uRegion, uSize, OutCtor);
  const sStorage = ArrayStorage.fromWasmRegion([k], outDtype, sRegion, sSize, OutCtor);
  const vtStorage = ArrayStorage.fromWasmRegion([n, n], outDtype, vtRegion, vtSize, OutCtor);

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
    length: number,
  ) => TypedArray;

  return ArrayStorage.fromWasmRegion([k], 'float64', sRegion, k, F64Ctor);
}

/**
 * WASM-accelerated one-sided Jacobi SVD for 2D complex matrices. Runs in the
 * precision of the input, so complex64 stays single throughout and comes back
 * complex64 with float32 singular values. n is capped at 256 by the kernel's
 * index array.
 *
 * @param a - Complex matrix
 * @returns { u, s, vt } with A = U diag(s) V^H, or null when WASM cannot take it
 */
export function wasmSvdComplex(
  a: ArrayStorage,
): { u: ArrayStorage; s: ArrayStorage; vt: ArrayStorage } | null {
  if (a.ndim !== 2) return null;
  const isC64 = a.dtype === 'complex64';
  if (a.dtype !== 'complex128' && !isC64) return null;

  const m = a.shape[0]!;
  const n = a.shape[1]!;
  if (n > 256) return null;
  if (
    m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier ||
    n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier
  )
    return null;

  const bpe = isC64 ? 4 : 8;
  const Arr = isC64 ? Float32Array : Float64Array;
  const k = Math.min(m, n);
  const uSlots = m * m * 2;
  const vtSlots = n * n * 2;
  const aSlots = m * n * 2;
  // Tail holds the n column norms; see the kernel's note on stack size.
  const workSlots = (m * n + n * n) * 2 + n;

  const regions = [];
  for (const slots of [uSlots, k, vtSlots, aSlots, workSlots]) {
    const region = wasmMalloc(slots * bpe);
    if (!region) {
      for (const r of regions) r.release();
      return null;
    }
    regions.push(region);
  }
  const [uRegion, sRegion, vtRegion, aRegion, workRegion] = regions;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const mem = getSharedMemory();
  const aView = new Arr(mem.buffer, aRegion!.ptr, aSlots);
  if (a.isCContiguous && !a.isWasmBacked) {
    aView.set((a.data as typeof aView).subarray(a.offset * 2, a.offset * 2 + aSlots));
  } else if (a.isCContiguous) {
    aView.set(new Arr(mem.buffer, a.wasmPtr + a.offset * 2 * bpe, aSlots));
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

  if (isC64) {
    svd_c64(aRegion!.ptr, uRegion!.ptr, sRegion!.ptr, vtRegion!.ptr, workRegion!.ptr, m, n);
  } else {
    svd_c128(aRegion!.ptr, uRegion!.ptr, sRegion!.ptr, vtRegion!.ptr, workRegion!.ptr, m, n);
  }
  workRegion!.release();
  aRegion!.release();

  const ctor = Arr as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number,
  ) => TypedArray;
  const cplx = isC64 ? 'complex64' : 'complex128';
  const real = isC64 ? 'float32' : 'float64';

  return {
    u: ArrayStorage.fromWasmRegion([m, m], cplx, uRegion!, uSlots, ctor),
    s: ArrayStorage.fromWasmRegion([k], real, sRegion!, k, ctor),
    vt: ArrayStorage.fromWasmRegion([n, n], cplx, vtRegion!, vtSlots, ctor),
  };
}
