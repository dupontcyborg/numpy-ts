/**
 * WASM-accelerated element-wise absolute value.
 *
 * Unary: out[i] = |a[i]|
 * Returns null if WASM can't handle this case.
 * Complex types are not handled here (magnitude needs sqrt, handled in JS).
 */

import { abs_f64, abs_f32, abs_f16, abs_i64, abs_i32, abs_i16, abs_i8 } from './bins/abs.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: abs_f64,
  float32: abs_f32,
  float16: abs_f16,
  int64: abs_i64,
  int32: abs_i32,
  int16: abs_i16,
  int8: abs_i8,
  // unsigned types: abs is identity, handled in JS
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: (typeof Float16Array !== 'undefined'
    ? Float16Array
    : Float32Array) as unknown as AnyTypedArrayCtor,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

/**
 * WASM-accelerated element-wise absolute value.
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmAbs(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  // Unsigned types: abs is identity — fast TypedArray copy (no WASM needed)
  if (dtype === 'uint8' || dtype === 'uint16' || dtype === 'uint32' || dtype === 'uint64') {
    const Ctor = ctorMap[dtype]!;
    const aOff = a.offset;
    const src = a.data.subarray(aOff, aOff + size) as TypedArray;
    const copy = new (Ctor as unknown as new (src: TypedArray) => TypedArray)(src);
    return ArrayStorage.fromData(copy, Array.from(a.shape), dtype);
  }

  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  kernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
