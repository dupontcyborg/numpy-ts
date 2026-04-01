/**
 * WASM-accelerated reduction product.
 *
 * Reduction: result = product(a[0..N])
 * Returns null if WASM can't handle this case.
 * uint types route to signed kernels (wrapping multiplication gives same bits).
 */

import {
  reduce_prod_f64,
  reduce_prod_f32,
  reduce_prod_i64,
  reduce_prod_i32,
  reduce_prod_i16,
  reduce_prod_i8,
  reduce_prod_u16,
  reduce_prod_u8,
  reduce_prod_strided_f64,
  reduce_prod_strided_f32,
  reduce_prod_strided_i64,
  reduce_prod_strided_i32,
  reduce_prod_strided_i16,
  reduce_prod_strided_i8,
  reduce_prod_strided_u64,
  reduce_prod_strided_u32,
  reduce_prod_strided_u16,
  reduce_prod_strided_u8,
} from './bins/reduce_prod.wasm';
import {
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  alloc,
  copyOut,
} from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  float64: reduce_prod_f64,
  float32: reduce_prod_f32,
  float16: reduce_prod_f32,
  int64: reduce_prod_i64,
  uint64: reduce_prod_i64,
  int32: reduce_prod_i32,
  uint32: reduce_prod_i32,
  // int8/int16 → i64, uint8/uint16 → u64 (promoted to avoid overflow)
  int16: reduce_prod_i16,
  uint16: reduce_prod_u16,
  int8: reduce_prod_i8,
  uint8: reduce_prod_u8,
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
 * WASM-accelerated reduction product (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceProd(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Float16: convert to f32, run f32 kernel, cast result back to f16 precision.
  // f32 accumulation + f16 cast matches NumPy's pairwise summation behavior.
  if (dtype === 'float16') {
    const aPtr = f16InputToScratchF32(a, size);
    const f32Result = Number(kernel(aPtr, size));
    // Cast through Float16Array to match f16 overflow (e.g., 65536 → inf)
    const f16Round = new Float16Array(1);
    f16Round[0] = f32Result;
    return f16Round[0]!;
  }

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  return Number(kernel(aPtr, size));
}

// --- Strided axis reduction for prod ---
// Output type depends on input: narrow ints promote, others stay native.

type StridedFn = (aPtr: number, outPtr: number, outer: number, axis: number, inner: number) => void;

const stridedKernels: Partial<Record<DType, StridedFn>> = {
  float64: reduce_prod_strided_f64,
  float32: reduce_prod_strided_f32,
  float16: reduce_prod_strided_f32,
  int64: reduce_prod_strided_i64,
  uint64: reduce_prod_strided_u64,
  int32: reduce_prod_strided_i32,
  uint32: reduce_prod_strided_u32,
  int16: reduce_prod_strided_i16,
  uint16: reduce_prod_strided_u16,
  int8: reduce_prod_strided_i8,
  uint8: reduce_prod_strided_u8,
};

// Output dtype: narrow ints promote to i64/u64, others stay native
const stridedOutDtype: Partial<Record<DType, DType>> = {
  float64: 'float64',
  float32: 'float32',
  float16: 'float16',
  int64: 'int64',
  uint64: 'uint64',
  int32: 'int32',
  uint32: 'uint32',
  int16: 'int64',
  uint16: 'uint64',
  int8: 'int64',
  uint8: 'uint64',
};

const stridedOutCtor: Partial<
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
  float16: Float32Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int64: BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint64: BigUint64Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int32: Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint32: Uint32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int16: BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint16: BigUint64Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int8: BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint8: BigUint64Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
};

const stridedOutBpe: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  float16: 4,
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 8,
  uint16: 8, // output is i64/u64 = 8 bytes
  int8: 8,
  uint8: 8, // output is i64/u64 = 8 bytes
};

/**
 * WASM-accelerated strided prod along an axis.
 * Output dtype may be promoted (narrow ints → i64/u64).
 * Returns output ArrayStorage, or null if WASM can't handle.
 */
export function wasmReduceProdStrided(
  a: ArrayStorage,
  outerSize: number,
  axisSize: number,
  innerSize: number
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const totalSize = outerSize * axisSize * innerSize;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = stridedKernels[dtype];
  const InCtor = ctorMap[dtype];
  const OutCtor = stridedOutCtor[dtype];
  const outDtype = stridedOutDtype[dtype];
  const outBpe = stridedOutBpe[dtype];
  if (!kernel || !InCtor || !OutCtor || !outDtype || !outBpe) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outSize = outerSize * innerSize;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let inPtr: number;
  if (dtype === 'float16') {
    inPtr = f16InputToScratchF32(a, totalSize);
  } else {
    inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, totalSize, inBpe);
  }
  const outPtr = alloc(outSize * outBpe);

  kernel(inPtr, outPtr, outerSize, axisSize, innerSize);

  const outData = copyOut(outPtr, outSize, OutCtor);

  // f16: cast f32 output back to f16 (matches NumPy overflow behavior)
  if (dtype === 'float16') {
    const f16Data = new Float16Array(outSize);
    f16Data.set(outData as Float32Array);
    return ArrayStorage.fromData(f16Data as unknown as TypedArray, [outSize], 'float16');
  }

  return ArrayStorage.fromData(outData, [outSize], outDtype);
}
