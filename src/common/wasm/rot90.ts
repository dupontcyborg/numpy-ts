/**
 * WASM-accelerated 2D 90-degree rotation (k=1, CCW).
 *
 * rot90: dst[r,c] = src[c, cols-1-r]
 * Returns null if WASM can't handle this case.
 */

import { rot90_f64, rot90_f32, rot90_i64, rot90_i32, rot90_i16, rot90_i8 } from './bins/rot90.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type Rot90Fn = (aPtr: number, outPtr: number, rows: number, cols: number) => void;

const kernels: Partial<Record<DType, Rot90Fn>> = {
  float64: rot90_f64,
  float32: rot90_f32,
  int64: rot90_i64,
  int32: rot90_i32,
  int16: rot90_i16,
  int8: rot90_i8,
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
};

/**
 * WASM-accelerated 2D rot90 (k=1, CCW).
 * Only handles 2D C-contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmRot90(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.shape.length !== 2) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const rows = a.shape[0]!;
  const cols = a.shape[1]!;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, rows, cols);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  // rot90 k=1: output shape is [cols, rows]
  return ArrayStorage.fromData(outData, [cols, rows], dtype);
}
