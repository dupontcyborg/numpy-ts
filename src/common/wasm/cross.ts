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
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
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
  float16: cross_f32,
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
  float16: Float32Array,
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
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const inputElements = batchSize * 3;
  const totalElements = inputElements * factor;
  const outBytes = totalElements * bpe;
  const isF16 = resultDtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    const aPtr = f16InputToScratchF32(a, totalElements);
    const bPtr = f16InputToScratchF32(b, totalElements);
    kernel(aPtr, bPtr, outRegion.ptr, batchSize);
    const f16Region = f32OutputToF16Region(outRegion, totalElements);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [...a.shape],
      resultDtype,
      f16Region,
      totalElements,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * factor,
    totalElements,
    bpe
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * factor,
    totalElements,
    bpe
  );

  kernel(aPtr, bPtr, outRegion.ptr, batchSize);

  return ArrayStorage.fromWasmRegion(
    [...a.shape],
    resultDtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
