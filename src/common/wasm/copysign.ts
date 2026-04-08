/**
 * WASM-accelerated element-wise copysign.
 *
 * Binary: out[i] = copysign(x1[i], x2[i])  (magnitude of x1, sign of x2)
 * Scalar: out[i] = copysign(x1[i], scalar)
 * Output is always float64. Returns null if WASM can't handle.
 */

import {
  copysign_f64,
  copysign_f32,
  copysign_i64,
  copysign_i32,
  copysign_i16,
  copysign_i8,
  copysign_u64,
  copysign_u32,
  copysign_u16,
  copysign_u8,
  copysign_scalar_f64,
  copysign_scalar_f32,
  copysign_scalar_i64,
  copysign_scalar_i32,
  copysign_scalar_i16,
  copysign_scalar_i8,
  copysign_scalar_u64,
  copysign_scalar_u32,
  copysign_scalar_u16,
  copysign_scalar_u8,
  copysign_f16,
  copysign_scalar_f16,
} from './bins/copysign.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, hasFloat16, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (x1Ptr: number, x2Ptr: number, outPtr: number, N: number) => void;
type ScalarFn = (x1Ptr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: copysign_f64,
  float32: copysign_f32,
  float16: copysign_f16,
  int64: copysign_i64,
  uint64: copysign_u64,
  int32: copysign_i32,
  uint32: copysign_u32,
  int16: copysign_i16,
  uint16: copysign_u16,
  int8: copysign_i8,
  uint8: copysign_u8,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: copysign_scalar_f64,
  float32: copysign_scalar_f32,
  float16: copysign_scalar_f16,
  int64: copysign_scalar_i64,
  uint64: copysign_scalar_u64,
  int32: copysign_scalar_i32,
  uint32: copysign_scalar_u32,
  int16: copysign_scalar_i16,
  uint16: copysign_scalar_u16,
  int8: copysign_scalar_i8,
  uint8: copysign_scalar_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const inputCtorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
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

// Output dtype: follows mathResultDtype (i8/u8→f16, i16/u16→f32, i32+→f64, floats preserve).
// WASM kernels output f32 for small ints; TS downcasts i8/u8 to f16.
const outDtypeMap: Partial<Record<DType, DType>> = {
  float64: 'float64',
  float32: 'float32',
  float16: 'float16',
  int64: 'float64',
  uint64: 'float64',
  int32: 'float64',
  uint32: 'float64',
  int16: 'float32',
  uint16: 'float32',
  int8: 'float32',
  uint8: 'float32',
};
const outCtorMap: Partial<
  Record<DType, new (buf: ArrayBuffer, off: number, len: number) => TypedArray>
> = {
  float64: Float64Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  float32: Float32Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  float16: (typeof Float16Array !== 'undefined' ? Float16Array : Float32Array) as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int64: Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint64: Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int32: Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint32: Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int16: Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint16: Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int8: Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint8: Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
};

/**
 * WASM-accelerated copysign of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmCopysign(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage | null {
  if (!x1.isCContiguous || !x2.isCContiguous) return null;

  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(x1.dtype);
  if (x2.dtype !== dtype) return null;

  const kernel = binaryKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  const outDtype = outDtypeMap[dtype];
  const OutCtor = outCtorMap[dtype];
  if (!kernel || !InCtor || !outDtype || !OutCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBpe = (OutCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  const outRegion = wasmMalloc(size * outBpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const x1Ptr = resolveInputPtr(x1.data, x1.isWasmBacked, x1.wasmPtr, x1.offset, size, inBpe);
  const x2Ptr = resolveInputPtr(x2.data, x2.isWasmBacked, x2.wasmPtr, x2.offset, size, inBpe);

  kernel(x1Ptr, x2Ptr, outRegion.ptr, size);

  // i8/u8 → downcast f32 to f16 (matches NumPy: int8/uint8 → float16)
  if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(x1.shape),
      'float16',
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return ArrayStorage.fromWasmRegion(Array.from(x1.shape), outDtype, outRegion, size, OutCtor);
}

/**
 * WASM-accelerated copysign of array and scalar.
 * Returns null if WASM can't handle.
 */
export function wasmCopysignScalar(x1: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!x1.isCContiguous) return null;

  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(x1.dtype);

  const kernel = scalarKernels[dtype];
  const InCtor = inputCtorMap[dtype];
  const outDtype = outDtypeMap[dtype];
  const OutCtor = outCtorMap[dtype];
  if (!kernel || !InCtor || !outDtype || !OutCtor) return null;

  const bpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBpe = (OutCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  const outRegion = wasmMalloc(size * outBpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const x1Ptr = resolveInputPtr(x1.data, x1.isWasmBacked, x1.wasmPtr, x1.offset, size, bpe);

  // Float16 scalar: pass sign flag (0 or 1) instead of raw scalar
  if (dtype === 'float16') {
    kernel(x1Ptr, outRegion.ptr, size, scalar < 0 || Object.is(scalar, -0) ? 1 : 0);
  } else {
    kernel(x1Ptr, outRegion.ptr, size, scalar);
  }

  // i8/u8 → downcast f32 to f16 (matches NumPy: int8/uint8 → float16)
  if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(x1.shape),
      'float16',
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return ArrayStorage.fromWasmRegion(Array.from(x1.shape), outDtype, outRegion, size, OutCtor);
}
