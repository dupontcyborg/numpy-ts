/**
 * WASM-accelerated batched 3-vector cross product.
 *
 * Computes n pairs of 3D cross products: out[i] = a[i] × b[i].
 * Returns null if WASM can't handle this case.
 */

import {
  cross_f64,
  cross_f32,
  cross_c128,
  cross_c64,
  cross_i64,
  cross_i32,
  cross_i16,
  cross_i8,
} from './bins/cross.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 8; // Minimum batch size for WASM

type WasmCrossFn = (a: number, b: number, out: number, n: number) => void;

const wasmKernels: Partial<Record<DType, WasmCrossFn>> = {
  float64: cross_f64,
  float32: cross_f32,
  complex128: cross_c128,
  complex64: cross_c64,
  int64: cross_i64,
  uint64: cross_i64,
  int32: cross_i32,
  uint32: cross_i32,
  int16: cross_i16,
  uint16: cross_i16,
  int8: cross_i8,
  uint8: cross_i8,
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
 * WASM-accelerated batched 3-vector cross product.
 * Both a and b must have shape [..., 3] with matching batch dimensions.
 * The vector axis must be the last axis and have length 3.
 * Returns ArrayStorage with same shape, or null.
 */
export function wasmCross(
  a: ArrayStorage,
  b: ArrayStorage,
  batchSize: number
): ArrayStorage | null {
  if (batchSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bytesPerElement = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = batchSize * 3;
  const aBytes = totalElements * factor * bytesPerElement;
  const bBytes = totalElements * factor * bytesPerElement;
  const outBytes = totalElements * factor * bytesPerElement;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aData = a.data.subarray(
    a.offset * factor,
    a.offset * factor + totalElements * factor
  ) as TypedArray;
  const bData = b.data.subarray(
    b.offset * factor,
    b.offset * factor + totalElements * factor
  ) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, batchSize);

  const outData = copyOut(
    outPtr,
    totalElements * factor,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, [...a.shape], resultDtype);
}
