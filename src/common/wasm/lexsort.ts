/**
 * WASM-accelerated lexsort (indirect stable sort on multiple keys).
 *
 * All keys must be the same dtype and contiguous.
 * Returns null if WASM can't handle this case.
 */

import {
  lexsort_f64,
  lexsort_f32,
  lexsort_i64,
  lexsort_u64,
  lexsort_i32,
  lexsort_u32,
  lexsort_i16,
  lexsort_u16,
  lexsort_i8,
  lexsort_u8,
} from './bins/lexsort.wasm';
import { ensureMemory, resetAllocator, alloc, copyOut, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type LexsortFn = (keysPtr: number, numKeys: number, N: number, outPtr: number) => void;

const kernels: Partial<Record<DType, LexsortFn>> = {
  float64: lexsort_f64,
  float32: lexsort_f32,
  int64: lexsort_i64,
  uint64: lexsort_u64,
  int32: lexsort_i32,
  uint32: lexsort_u32,
  int16: lexsort_i16,
  uint16: lexsort_u16,
  int8: lexsort_i8,
  uint8: lexsort_u8,
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
 * WASM-accelerated lexsort.
 * All keys must be the same dtype and 1D contiguous.
 * Returns ArrayStorage of int32 indices or null if WASM can't handle it.
 */
export function wasmLexsort(keys: ArrayStorage[]): ArrayStorage | null {
  if (keys.length === 0) return null;

  const n = keys[0]!.size;
  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // All keys must be same dtype and contiguous
  const dtype = keys[0]!.dtype;
  for (const key of keys) {
    if (key.dtype !== dtype || !key.isCContiguous || key.ndim !== 1 || key.size !== n) {
      return null;
    }
  }

  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const numKeys = keys.length;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const keysBytes = numKeys * n * bpe;
  const outBytes = n * 4; // u32 indices

  ensureMemory(keysBytes + outBytes);
  resetAllocator();

  // Copy all keys into a single flat buffer: keys[0] || keys[1] || ...
  const flatBuf = alloc(keysBytes);
  const mem = getSharedMemory();

  // Actually, we need to use copyIn but for a flat concatenated buffer.
  // Let's manually copy each key in sequence.
  for (let k = 0; k < numKeys; k++) {
    const key = keys[k]!;
    const kOff = key.offset;
    const kData = key.data.subarray(kOff, kOff + n) as TypedArray;
    const destOffset = flatBuf + k * n * bpe;
    new Uint8Array(mem.buffer, destOffset, n * bpe).set(
      new Uint8Array(kData.buffer, kData.byteOffset, kData.byteLength)
    );
  }

  const outPtr = alloc(outBytes);

  kernel(flatBuf, numKeys, n, outPtr);

  const outData = copyOut(
    outPtr,
    n,
    Int32Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => Int32Array
  );

  return ArrayStorage.fromData(outData, [n], 'int32');
}
