/**
 * WASM-accelerated flat repeat.
 *
 * repeat: Each element a[i] is written `reps` times to output.
 * Returns null if WASM can't handle this case.
 */

import {
  repeat_f64,
  repeat_f32,
  repeat_i64,
  repeat_i32,
  repeat_i16,
  repeat_i8,
} from './bins/repeat.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type RepeatFn = (aPtr: number, outPtr: number, N: number, reps: number) => void;

const kernels: Partial<Record<DType, RepeatFn>> = {
  float64: repeat_f64,
  float32: repeat_f32,
  int64: repeat_i64,
  uint64: repeat_i64,
  int32: repeat_i32,
  uint32: repeat_i32,
  int16: repeat_i16,
  uint16: repeat_i16,
  int8: repeat_i8,
  uint8: repeat_i8,
  float16: repeat_i16, // byte-copy: treat f16 as raw i16
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
 * WASM-accelerated flat repeat with uniform repeat count.
 * Only handles flattened repeat (no axis) with a single repeat count (not array).
 * Returns null if WASM can't handle.
 */
export function wasmRepeat(a: ArrayStorage, reps: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const outSize = size * reps;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = outSize * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 2);
    kernel(aPtr, outRegion.ptr, size, reps);
    return ArrayStorage.fromWasmRegion(
      [outSize],
      dtype,
      outRegion,
      outSize,
      Float16Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, size, reps);

  return ArrayStorage.fromWasmRegion(
    [outSize],
    dtype,
    outRegion,
    outSize,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
