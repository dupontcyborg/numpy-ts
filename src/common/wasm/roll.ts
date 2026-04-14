/**
 * WASM-accelerated flat array roll (circular shift).
 *
 * roll: out[i] = a[(i - shift + N) % N]
 * Returns null if WASM can't handle this case.
 */

import { roll_f64, roll_f32, roll_i64, roll_i32, roll_i16, roll_i8 } from './bins/roll.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type RollFn = (aPtr: number, outPtr: number, N: number, shift: number) => void;

const kernels: Partial<Record<DType, RollFn>> = {
  float64: roll_f64,
  float32: roll_f32,
  int64: roll_i64,
  uint64: roll_i64,
  int32: roll_i32,
  uint32: roll_i32,
  int16: roll_i16,
  uint16: roll_i16,
  int8: roll_i8,
  uint8: roll_i8,
  float16: roll_i16, // byte-copy: treat f16 as raw i16
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
  float16: (typeof Float16Array !== 'undefined'
    ? Float16Array
    : Float32Array) as unknown as AnyTypedArrayCtor,
};

/**
 * WASM-accelerated flat roll (no axis).
 * Returns null if WASM can't handle.
 */
export function wasmRoll(a: ArrayStorage, shift: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Float16: use i16 kernel on raw bytes (byte-copy, no conversion needed)
  if (isF16) {
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 2);
    kernel(aPtr, outRegion.ptr, size, shift);
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Float16Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, size, shift);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
