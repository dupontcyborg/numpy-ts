/**
 * WASM-accelerated argwhere — find flat indices of non-zero elements.
 *
 * Returns a Uint32Array of flat indices, or null if WASM can't handle this case.
 * The TS ops layer converts flat indices to multi-indices.
 */

import {
  argwhere_fill_f64,
  argwhere_fill_f32,
  argwhere_fill_i64,
  argwhere_fill_i32,
  argwhere_fill_i16,
  argwhere_fill_i8,
} from './bins/argwhere.wasm';
import {
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  getSharedMemory,
  f16InputToScratchF32,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type FillFn = (aPtr: number, outPtr: number, N: number) => number;

const fillKernels: Partial<Record<DType, FillFn>> = {
  float64: argwhere_fill_f64,
  float32: argwhere_fill_f32,
  float16: argwhere_fill_f32,
  int64: argwhere_fill_i64,
  uint64: argwhere_fill_i64,
  int32: argwhere_fill_i32,
  uint32: argwhere_fill_i32,
  int16: argwhere_fill_i16,
  uint16: argwhere_fill_i16,
  int8: argwhere_fill_i8,
  uint8: argwhere_fill_i8,
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
 * WASM-accelerated argwhere for contiguous arrays.
 * Returns a Uint32Array of flat indices of non-zero elements, or null.
 */
export function wasmArgwhereFlat(a: ArrayStorage): Uint32Array | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = fillKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Resolve input pointer
  let aPtr: number;
  if (dtype === 'float16') {
    aPtr = f16InputToScratchF32(a, size);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  }

  // Allocate worst-case output (N u32 indices) in scratch
  const outBytes = size * 4; // u32 = 4 bytes
  const outPtr = scratchAlloc(outBytes);

  // Fill indices
  const count = kernel(aPtr, outPtr, size);

  // Copy result out of WASM memory
  const mem = getSharedMemory();
  const wasmView = new Uint32Array(mem.buffer, outPtr, count);
  const result = new Uint32Array(count);
  result.set(wasmView);

  return result;
}
