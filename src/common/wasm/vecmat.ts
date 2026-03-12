/**
 * WASM-accelerated vector-matrix product.
 *
 * Computes y[j] = sum_k x[k] * A[k,j] for x[K] and A[K,N].
 * Returns null if WASM can't handle this case.
 */

import {
  vecmat_f64,
  vecmat_f32,
  vecmat_c128,
  vecmat_c64,
  vecmat_i64,
  vecmat_i32,
  vecmat_i16,
  vecmat_i8,
} from './bins/vecmat.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 128; // Minimum K*N for WASM

type WasmVecmatFn = (x: number, A: number, y: number, K: number, N: number) => void;

const wasmKernels: Partial<Record<DType, WasmVecmatFn>> = {
  float64: vecmat_f64,
  float32: vecmat_f32,
  complex128: vecmat_c128,
  complex64: vecmat_c64,
  int64: vecmat_i64,
  uint64: vecmat_i64,
  int32: vecmat_i32,
  uint32: vecmat_i32,
  int16: vecmat_i16,
  uint16: vecmat_i16,
  int8: vecmat_i8,
  uint8: vecmat_i8,
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
 * WASM-accelerated vecmat: x[K] · A[K,N] → y[N].
 * x must be 1D, A must be 2D, both contiguous.
 */
export function wasmVecmat(x: ArrayStorage, A: ArrayStorage): ArrayStorage | null {
  if (x.ndim !== 1 || A.ndim !== 2) return null;
  if (!x.isCContiguous || !A.isCContiguous) return null;

  const K = A.shape[0]!;
  const N = A.shape[1]!;
  if (K !== x.shape[0]!) return null;
  if (K * N < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = promoteDTypes(x.dtype, A.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bytesPerElement = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const xBytes = K * factor * bytesPerElement;
  const aBytes = K * N * factor * bytesPerElement;
  const outBytes = N * factor * bytesPerElement;

  ensureMemory(xBytes + aBytes + outBytes);
  resetAllocator();

  const xData = x.data.subarray(x.offset * factor, x.offset * factor + K * factor) as TypedArray;
  const aData = A.data.subarray(
    A.offset * factor,
    A.offset * factor + K * N * factor
  ) as TypedArray;

  const xPtr = copyIn(xData);
  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(xPtr, aPtr, outPtr, K, N);

  const outData = copyOut(
    outPtr,
    N * factor,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, [N], resultDtype);
}
