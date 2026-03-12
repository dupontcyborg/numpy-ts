/**
 * WASM-accelerated element-wise clip (clamp).
 *
 * Unary: out[i] = clamp(a[i], lo, hi)
 * Returns null if WASM can't handle this case.
 */

import {
  clip_f64, clip_f32, clip_i64, clip_i32, clip_i16, clip_i8,
  clip_u64, clip_u32, clip_u16, clip_u8,
} from './bins/clip.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ClipFn = (aPtr: number, outPtr: number, N: number, lo: number, hi: number) => void;

const kernels: Partial<Record<DType, ClipFn>> = {
  float64: clip_f64, float32: clip_f32,
  int64: clip_i64, uint64: clip_u64,
  int32: clip_i32, uint32: clip_u32,
  int16: clip_i16, uint16: clip_u16,
  int8: clip_i8, uint8: clip_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array, float32: Float32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
  int32: Int32Array, uint32: Uint32Array,
  int16: Int16Array, uint16: Uint16Array,
  int8: Int8Array, uint8: Uint8Array,
};

/**
 * WASM-accelerated clip with scalar bounds.
 * Returns null if WASM can't handle (complex, non-contiguous, array bounds, too small).
 */
export function wasmClip(a: ArrayStorage, lo: number, hi: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 2);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(aPtr, outPtr, size, lo, hi);

  const outData = copyOut(outPtr, size, Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray);
  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
