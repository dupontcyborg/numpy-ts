/**
 * WASM-accelerated element-wise minimum.
 *
 * Binary: out[i] = min(a[i], b[i])
 * Scalar: out[i] = min(a[i], scalar)
 * Returns null if WASM can't handle this case.
 */

import {
  min_f64,
  min_f32,
  min_i64,
  min_i32,
  min_i16,
  min_i8,
  min_scalar_f64,
  min_scalar_f32,
  min_scalar_i64,
  min_scalar_i32,
  min_scalar_i16,
  min_scalar_i8,
  min_u64,
  min_u32,
  min_u16,
  min_u8,
  min_scalar_u64,
  min_scalar_u32,
  min_scalar_u16,
  min_scalar_u8,
} from './bins/min.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: min_f64,
  float32: min_f32,
  int64: min_i64,
  uint64: min_u64,
  int32: min_i32,
  uint32: min_u32,
  int16: min_i16,
  uint16: min_u16,
  int8: min_i8,
  uint8: min_u8,
  float16: min_f32,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: min_scalar_f64,
  float32: min_scalar_f32,
  int64: min_scalar_i64,
  uint64: min_scalar_u64,
  int32: min_scalar_i32,
  uint32: min_scalar_u32,
  int16: min_scalar_i16,
  uint16: min_scalar_u16,
  int8: min_scalar_i8,
  uint8: min_scalar_u8,
  float16: min_scalar_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

export function wasmMin(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;
  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  const originalDtype = dtype;
  resetScratchAllocator();

  let aPtr: number;
  let bPtr: number;
  if (dtype === 'float16') {
    aPtr = f16InputToScratchF32(a, size);
    bPtr = f16InputToScratchF32(b, size);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);
  }

  kernel(aPtr, bPtr, outRegion.ptr, size);

  if (originalDtype === 'float16') {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      originalDtype,
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}

export function wasmMinScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();

  let aPtr: number;
  if (dtype === 'float16') {
    aPtr = f16InputToScratchF32(a, size);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  }

  kernel(aPtr, outRegion.ptr, size, scalar);

  if (dtype === 'float16') {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
