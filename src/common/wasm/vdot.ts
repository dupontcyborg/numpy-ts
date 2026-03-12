/**
 * WASM-accelerated conjugate dot product (vdot) for complex types.
 *
 * Computes sum_k conj(a[k]) * b[k] for flattened 1D complex arrays.
 * For real/integer types, vdot = dot — use wasmDot1D instead.
 * Returns null if WASM can't handle this case.
 */

import { vdot_c128, vdot_c64 } from './bins/vdot.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { Complex } from '../complex';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 64; // Minimum K for WASM

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
  const aBytes = K * 2 * bytesPerElement;
  const bBytes = K * 2 * bytesPerElement;
  const outBytes = 2 * bytesPerElement;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aData = a.data.subarray(a.offset * 2, a.offset * 2 + K * 2) as TypedArray;
  const bData = b.data.subarray(b.offset * 2, b.offset * 2 + K * 2) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, K);

  const outData = copyOut(
    outPtr,
    2,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return new Complex(Number(outData[0]!), Number(outData[1]!));
}
