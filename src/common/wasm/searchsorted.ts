/**
 * WASM-accelerated searchsorted (binary search for insertion indices).
 *
 * Output is float64 indices for JS ergonomics (no BigInt).
 * WASM searches u32 internally and converts to f64 before returning.
 * Returns null if WASM can't handle this case.
 */

import {
  searchsorted_left_f64,
  searchsorted_right_f64,
  searchsorted_left_f32,
  searchsorted_right_f32,
  searchsorted_left_i64,
  searchsorted_right_i64,
  searchsorted_left_u64,
  searchsorted_right_u64,
  searchsorted_left_i32,
  searchsorted_right_i32,
  searchsorted_left_u32,
  searchsorted_right_u32,
  searchsorted_left_i16,
  searchsorted_right_i16,
  searchsorted_left_u16,
  searchsorted_right_u16,
  searchsorted_left_i8,
  searchsorted_right_i8,
  searchsorted_left_u8,
  searchsorted_right_u8,
} from './bins/searchsorted.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type SearchFn = (
  sortedPtr: number,
  N: number,
  valuesPtr: number,
  outPtr: number,
  M: number
) => void;

const leftKernels: Partial<Record<DType, SearchFn>> = {
  float64: searchsorted_left_f64,
  float32: searchsorted_left_f32,
  int64: searchsorted_left_i64,
  uint64: searchsorted_left_u64,
  int32: searchsorted_left_i32,
  uint32: searchsorted_left_u32,
  int16: searchsorted_left_i16,
  uint16: searchsorted_left_u16,
  int8: searchsorted_left_i8,
  uint8: searchsorted_left_u8,
  float16: searchsorted_left_f32,
};

const rightKernels: Partial<Record<DType, SearchFn>> = {
  float64: searchsorted_right_f64,
  float32: searchsorted_right_f32,
  int64: searchsorted_right_i64,
  uint64: searchsorted_right_u64,
  int32: searchsorted_right_i32,
  uint32: searchsorted_right_u32,
  int16: searchsorted_right_i16,
  uint16: searchsorted_right_u16,
  int8: searchsorted_right_i8,
  uint8: searchsorted_right_u8,
  float16: searchsorted_right_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
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

/**
 * WASM-accelerated searchsorted.
 * Returns ArrayStorage of float64 insertion indices or null if WASM can't handle it.
 */
export function wasmSearchsorted(
  sorted: ArrayStorage,
  values: ArrayStorage,
  side: 'left' | 'right'
): ArrayStorage | null {
  if (!sorted.isCContiguous || !values.isCContiguous) return null;

  const n = sorted.size;
  const m = values.size;
  if (m < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(sorted.dtype);
  const kernelMap = side === 'left' ? leftKernels : rightKernels;
  const kernel = kernelMap[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  // Values must be same dtype
  if (values.dtype !== dtype) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = m * 8; // f64 indices

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const isF16 = dtype === 'float16';

  let sortedPtr: number;
  let valuesPtr: number;

  if (isF16) {
    sortedPtr = f16InputToScratchF32(sorted, n);
    valuesPtr = f16InputToScratchF32(values, m);
  } else {
    sortedPtr = resolveInputPtr(
      sorted.data,
      sorted.isWasmBacked,
      sorted.wasmPtr,
      sorted.offset,
      n,
      bpe
    );
    valuesPtr = resolveInputPtr(
      values.data,
      values.isWasmBacked,
      values.wasmPtr,
      values.offset,
      m,
      bpe
    );
  }

  kernel(sortedPtr, n, valuesPtr, outRegion.ptr, m);

  return ArrayStorage.fromWasmRegion(
    Array.from(values.shape),
    'float64',
    outRegion,
    m,
    Float64Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
