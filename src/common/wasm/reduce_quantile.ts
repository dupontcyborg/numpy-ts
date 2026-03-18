/**
 * WASM-accelerated quantile computation.
 *
 * Sorts a copy of the data in WASM, then interpolates.
 * All dtypes are converted to f64 before passing to WASM.
 * Returns null if WASM can't handle this case.
 */

import { reduce_quantile_f64 } from './bins/reduce_quantile.wasm';
import { ensureMemory, resetAllocator, copyIn } from './runtime';
import { ArrayStorage } from '../storage';
import { isBigIntDType, isComplexDType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

/**
 * WASM-accelerated quantile (no axis, full array).
 * Converts all data to f64, sorts in WASM, interpolates.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmReduceQuantile(a: ArrayStorage, q: number): number | null {
  if (!a.isCContiguous) return null;
  if (isComplexDType(a.dtype)) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Convert to f64 in JS, then copy to WASM
  const f64Buf = new Float64Array(size);
  const off = a.offset;
  const data = a.data;

  if (a.dtype === 'float64') {
    f64Buf.set((data as Float64Array).subarray(off, off + size));
  } else if (isBigIntDType(a.dtype)) {
    const typed = data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) f64Buf[i] = Number(typed[off + i]!);
  } else {
    for (let i = 0; i < size; i++) f64Buf[i] = Number(data[off + i]!);
  }

  ensureMemory(size * 8);
  resetAllocator();

  const ptr = copyIn(f64Buf as unknown as TypedArray);

  // Sort in-place in WASM and compute quantile
  return reduce_quantile_f64(ptr, size, q);
}
