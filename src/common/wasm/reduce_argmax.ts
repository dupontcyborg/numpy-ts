/**
 * WASM-accelerated reduction argmax.
 *
 * Reduction: result = argmax(a[0..N])
 * Returns null if WASM can't handle this case.
 * Unsigned types use SEPARATE kernels.
 */

import { type DType, effectiveDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  argmax_axis_f32,
  argmax_axis_f64,
  argmax_axis_i8,
  argmax_axis_i16,
  argmax_axis_i32,
  argmax_axis_i64,
  argmax_axis_u8,
  argmax_axis_u16,
  argmax_axis_u32,
  argmax_axis_u64,
  reduce_argmax_f32,
  reduce_argmax_f64,
  reduce_argmax_i8,
  reduce_argmax_i16,
  reduce_argmax_i32,
  reduce_argmax_i64,
  reduce_argmax_strided_f32,
  reduce_argmax_strided_f64,
  reduce_argmax_strided_i8,
  reduce_argmax_strided_i16,
  reduce_argmax_strided_i32,
  reduce_argmax_strided_u8,
  reduce_argmax_strided_u16,
  reduce_argmax_strided_u32,
  reduce_argmax_u8,
  reduce_argmax_u16,
  reduce_argmax_u32,
  reduce_argmax_u64,
} from './bins/reduce_argmax.wasm';
import { wasmConfig } from './config';
import {
  f16InputToScratchF32,
  resetScratchAllocator,
  resolveInputPtr,
  wasmMalloc,
} from './runtime';

const BASE_THRESHOLD = 32;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_argmax_f64,
  float32: reduce_argmax_f32,
  float16: reduce_argmax_f32,
  int64: reduce_argmax_i64,
  uint64: reduce_argmax_u64,
  int32: reduce_argmax_i32,
  uint32: reduce_argmax_u32,
  int16: reduce_argmax_i16,
  uint16: reduce_argmax_u16,
  int8: reduce_argmax_i8,
  uint8: reduce_argmax_u8,
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

/**
 * WASM-accelerated reduction argmax (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceArgmax(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let aPtr: number;
  if (dtype === 'float16') {
    aPtr = f16InputToScratchF32(a, size);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  }

  return Number(kernel(aPtr, size));
}

// --- Strided axis reduction (output is always int32) ---

type StridedFn = (aPtr: number, outPtr: number, outer: number, axis: number, inner: number) => void;

const stridedKernels: Partial<Record<DType, StridedFn>> = {
  float64: reduce_argmax_strided_f64,
  float32: reduce_argmax_strided_f32,
  float16: reduce_argmax_strided_f32,
  int32: reduce_argmax_strided_i32,
  uint32: reduce_argmax_strided_u32,
  int16: reduce_argmax_strided_i16,
  uint16: reduce_argmax_strided_u16,
  int8: reduce_argmax_strided_i8,
  uint8: reduce_argmax_strided_u8,
};

/**
 * WASM-accelerated strided argmax along an axis.
 * Output dtype is always int32 (indices). Returns output ArrayStorage, or null if WASM can't handle.
 */
export function wasmReduceArgmaxStrided(
  a: ArrayStorage,
  outerSize: number,
  axisSize: number,
  innerSize: number,
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const totalSize = outerSize * axisSize * innerSize;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = stridedKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!kernel || !InCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outSize = outerSize * innerSize;

  const outRegion = wasmMalloc(outSize * 4);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let inPtr: number;
  if (dtype === 'float16') {
    inPtr = f16InputToScratchF32(a, totalSize);
  } else {
    inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, totalSize, inBpe);
  }

  kernel(inPtr, outRegion.ptr, outerSize, axisSize, innerSize);

  return ArrayStorage.fromWasmRegion(
    [outSize],
    'int32',
    outRegion,
    outSize,
    Int32Array as unknown as new (
      buf: ArrayBuffer,
      off: number,
      len: number,
    ) => TypedArray,
  );
}

// --- Strided reduction along any single axis of a C-contiguous array. ---
// Output is int64 (matching NumPy). before = ∏ dims[0..axis], axisSize =
// dims[axis], inner = ∏ dims[axis+1..] (the axis stride). axis=0 keeps the
// original column-scan access (before=1); the last axis becomes a contiguous
// scan (inner=1).

type AxisFn = (
  aPtr: number,
  before: number,
  axisSize: number,
  inner: number,
  outPtr: number,
) => void;

const axisKernels: Partial<Record<DType, AxisFn>> = {
  float64: argmax_axis_f64,
  float32: argmax_axis_f32,
  float16: argmax_axis_f32,
  int64: argmax_axis_i64,
  uint64: argmax_axis_u64,
  int32: argmax_axis_i32,
  uint32: argmax_axis_u32,
  int16: argmax_axis_i16,
  uint16: argmax_axis_u16,
  int8: argmax_axis_i8,
  uint8: argmax_axis_u8,
};

/**
 * WASM-accelerated argmax along any single axis of a C-contiguous array.
 * Output is int64 (matching NumPy). Returns null if WASM can't handle.
 */
export function wasmArgmaxAxis(a: ArrayStorage, axis: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const ndim = a.shape.length;
  if (ndim === 0 || axis < 0 || axis >= ndim) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = axisKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!kernel || !InCtor) return null;

  const shape = a.shape;
  let before = 1;
  for (let i = 0; i < axis; i++) before *= shape[i]!;
  const axisSize = shape[axis]!;
  let inner = 1;
  for (let i = axis + 1; i < ndim; i++) inner *= shape[i]!;
  const outLen = before * inner;
  const outShape = [...shape.slice(0, axis), ...shape.slice(axis + 1)];

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outRegion = wasmMalloc(outLen * 8);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let inPtr: number;
  if (dtype === 'float16') {
    inPtr = f16InputToScratchF32(a, size);
  } else {
    inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);
  }

  kernel(inPtr, before, axisSize, inner, outRegion.ptr);

  return ArrayStorage.fromWasmRegion(
    outShape,
    'int64',
    outRegion,
    outLen,
    BigInt64Array as unknown as new (
      buf: ArrayBuffer,
      off: number,
      len: number,
    ) => TypedArray,
  );
}
