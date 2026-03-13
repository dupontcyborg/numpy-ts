/**
 * WASM-accelerated flat array roll (circular shift).
 *
 * roll: out[i] = a[(i - shift + N) % N]
 * Returns null if WASM can't handle this case.
 */

import { roll_f64, roll_f32, roll_i64, roll_i32, roll_i16, roll_i8 } from './bins/roll.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type RollFn = (aPtr: number, outPtr: number, N: number, shift: number) => void;

const kernels: Partial<Record<DType, RollFn>> = {
  float64: roll_f64,
  float32: roll_f32,
  int64: roll_i64,
  int32: roll_i32,
  int16: roll_i16,
  int8: roll_i8,
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
 * WASM-accelerated flat roll (no axis).
 * Returns null if WASM can't handle.
 */
export function wasmRoll(a: ArrayStorage, shift: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size, shift);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
