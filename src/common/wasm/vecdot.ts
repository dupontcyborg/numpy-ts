/**
 * WASM-accelerated batched vector dot product (vecdot).
 *
 * Computes out[i] = sum_k a[i,k] * b[i,k] for contiguous 2D arrays
 * where the last axis is contracted. Both a and b must have the same shape.
 * Returns null if WASM can't handle this case.
 */

import {
  vecdot_f64,
  vecdot_f32,
  vecdot_c128,
  vecdot_c64,
  vecdot_i64,
  vecdot_i32,
  vecdot_i16,
  vecdot_i8,
} from './bins/vecdot.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 256; // Minimum B*K for WASM

type WasmVecdotFn = (a: number, b: number, out: number, B: number, K: number) => void;

const wasmKernels: Partial<Record<DType, WasmVecdotFn>> = {
  float64: vecdot_f64,
  float32: vecdot_f32,
  complex128: vecdot_c128,
  complex64: vecdot_c64,
  int64: vecdot_i64,
  uint64: vecdot_i64,
  int32: vecdot_i32,
  uint32: vecdot_i32,
  int16: vecdot_i16,
  uint16: vecdot_i16,
  int8: vecdot_i8,
  uint8: vecdot_i8,
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
 * WASM-accelerated vecdot for 2D arrays with contraction along the last axis.
 * a and b must be 2D with matching shapes, both C-contiguous.
 * Returns ArrayStorage with shape [B] (batch dimension), or null.
 */
export function wasmVecdot(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  // Only handle 2D case where both have same shape and last axis is contracted
  if (a.ndim !== 2 || b.ndim !== 2) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const B = a.shape[0]!;
  const K = a.shape[1]!;
  if (B !== b.shape[0]! || K !== b.shape[1]!) return null;
  if (B * K < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bytesPerElement = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = B * K * factor * bytesPerElement;
  const bBytes = B * K * factor * bytesPerElement;
  const outBytes = B * factor * bytesPerElement;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aData = a.data.subarray(
    a.offset * factor,
    a.offset * factor + B * K * factor
  ) as TypedArray;
  const bData = b.data.subarray(
    b.offset * factor,
    b.offset * factor + B * K * factor
  ) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, B, K);

  const outData = copyOut(
    outPtr,
    B * factor,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, [B], resultDtype);
}
