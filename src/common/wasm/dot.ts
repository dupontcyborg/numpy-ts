/**
 * WASM-accelerated 1D dot product.
 *
 * Computes sum_k a[k] * b[k] for contiguous 1D arrays.
 * Returns null if WASM can't handle this case.
 */

import {
  dot_f64,
  dot_f32,
  dot_c128,
  dot_c64,
  dot_i64,
  dot_i32,
  dot_i16,
  dot_i8,
} from './bins/dot.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { Complex } from '../complex';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 128; // Minimum K for WASM to be worth it

type WasmDotFn = (aPtr: number, bPtr: number, outPtr: number, K: number) => void;

const wasmKernels: Partial<Record<DType, WasmDotFn>> = {
  float64: dot_f64,
  float32: dot_f32,
  complex128: dot_c128,
  complex64: dot_c64,
  int64: dot_i64,
  uint64: dot_i64,
  int32: dot_i32,
  uint32: dot_i32,
  int16: dot_i16,
  uint16: dot_i16,
  int8: dot_i8,
  uint8: dot_i8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  complex128: Float64Array,
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated 1D dot product. Returns null if WASM can't handle.
 * Both a and b must be 1D, contiguous, same-length.
 */
export function wasmDot1D(a: ArrayStorage, b: ArrayStorage): number | Complex | null {
  if (a.ndim !== 1 || b.ndim !== 1) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const K = a.shape[0]!;
  if (K !== b.shape[0]!) return null;
  if (K < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;

  const bytesPerElement = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = K * factor * bytesPerElement;
  const bBytes = K * factor * bytesPerElement;
  const outBytes = 1 * factor * bytesPerElement;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aData = a.data.subarray(a.offset * factor, a.offset * factor + K * factor) as TypedArray;
  const bData = b.data.subarray(b.offset * factor, b.offset * factor + K * factor) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, K);

  const outData = copyOut(
    outPtr,
    1 * factor,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  // Complex scalar: read re + im from the 2-element output buffer
  if (factor === 2) {
    return new Complex(
      Number((outData as Float64Array | Float32Array)[0]!),
      Number((outData as Float64Array | Float32Array)[1]!)
    );
  }

  return Number(outData[0]!);
}
