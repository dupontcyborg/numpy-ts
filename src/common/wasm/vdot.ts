/**
 * WASM-accelerated conjugate dot product (vdot) for complex types.
 *
 * Computes sum_k conj(a[k]) * b[k] for flattened 1D complex arrays.
 * For real/integer types, vdot = dot — use wasmDot1D instead.
 * Returns null if WASM can't handle this case.
 */

import { vdot_c128, vdot_c64 } from './bins/vdot.wasm';
import { resetScratchAllocator, resolveInputPtr, scratchAlloc, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { Complex } from '../complex';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 32; // Minimum K for WASM

type WasmVdotFn = (a: number, b: number, out: number, K: number) => void;

const wasmKernels: Partial<Record<DType, WasmVdotFn>> = {
  complex128: vdot_c128,
  complex64: vdot_c64,
};

const ctorMap: Partial<Record<DType, new (length: number) => TypedArray>> = {
  complex128: Float64Array,
  complex64: Float32Array,
};

/**
 * WASM-accelerated conjugate dot product for complex types.
 * Both a and b must be 1D, contiguous, same-length, complex dtype.
 * Returns Complex or null.
 */
export function wasmVdotComplex(a: ArrayStorage, b: ArrayStorage): Complex | null {
  if (a.ndim !== 1 || b.ndim !== 1) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const K = a.shape[0]!;
  if (K !== b.shape[0]!) return null;
  if (K < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const bytesPerElement = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = 2 * bytesPerElement;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * 2,
    K * 2,
    bytesPerElement
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * 2,
    K * 2,
    bytesPerElement
  );
  const outPtr = scratchAlloc(outBytes);

  kernel(aPtr, bPtr, outPtr, K);

  // Read scalar result directly from WASM memory
  const mem = getSharedMemory();
  const outView = new (Ctor as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number
  ) => TypedArray)(mem.buffer, outPtr, 2);

  return new Complex(Number(outView[0]!), Number(outView[1]!));
}
