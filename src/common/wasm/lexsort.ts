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
import {
  wasmMalloc,
  resetScratchAllocator,
  scratchAlloc,
  getSharedMemory,
  f16InputToScratchF32,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type LexsortFn = (keysPtr: number, numKeys: number, N: number, outPtr: number) => void;
type LexsortRadixFn = (
  keysPtr: number,
  numKeys: number,
  N: number,
  outPtr: number,
  scratchPtr: number
) => void;

const kernels: Partial<Record<DType, LexsortFn>> = {
  float64: lexsort_f64,
  float32: lexsort_f32,
  int64: lexsort_i64,
  uint64: lexsort_u64,
  int32: lexsort_i32,
  uint32: lexsort_u32,
  float16: lexsort_f32,
};

// Narrow integer types use radix sort (extra scratch parameter)
const radixKernels: Partial<Record<DType, LexsortRadixFn>> = {
  int16: lexsort_i16 as unknown as LexsortRadixFn,
  uint16: lexsort_u16 as unknown as LexsortRadixFn,
  int8: lexsort_i8 as unknown as LexsortRadixFn,
  uint8: lexsort_u8 as unknown as LexsortRadixFn,
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
 * WASM-accelerated lexsort.
 * All keys must be the same dtype and 1D contiguous.
 * Returns ArrayStorage of int32 indices or null if WASM can't handle it.
 */
export function wasmLexsort(keys: ArrayStorage[]): ArrayStorage | null {
  if (keys.length === 0) return null;

  const n = keys[0]!.size;
  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // All keys must be same dtype and contiguous
  const dtype = effectiveDType(keys[0]!.dtype);
  for (const key of keys) {
    if (key.dtype !== dtype || !key.isCContiguous || key.ndim !== 1 || key.size !== n) {
      return null;
    }
  }

  const kernel = kernels[dtype];
  const radixKernel = radixKernels[dtype];
  const Ctor = ctorMap[dtype];
  if ((!kernel && !radixKernel) || !Ctor) return null;

  const numKeys = keys.length;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const keysBytes = numKeys * n * bpe;
  const outBytes = n * 4; // i32 indices

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Allocate one flat buffer for all keys concatenated (no alignment gaps)
  const isF16 = dtype === 'float16';
  let flatBuf: number;
  if (isF16) {
    // f16→f32: each key converted via bump allocator (contiguous since all same size)
    flatBuf = 0;
    for (let k = 0; k < numKeys; k++) {
      const ptr = f16InputToScratchF32(keys[k]!, n);
      if (k === 0) flatBuf = ptr;
    }
  } else {
    flatBuf = scratchAlloc(keysBytes);
    const mem = getSharedMemory();
    for (let k = 0; k < numKeys; k++) {
      const key = keys[k]!;
      const kData = key.data.subarray(key.offset, key.offset + n) as TypedArray;
      const destOffset = flatBuf + k * n * bpe;
      new Uint8Array(mem.buffer, destOffset, n * bpe).set(
        new Uint8Array(kData.buffer, kData.byteOffset, kData.byteLength)
      );
    }
  }

  if (radixKernel) {
    const scratchPtr = scratchAlloc(n * 4);
    radixKernel(flatBuf, numKeys, n, outRegion.ptr, scratchPtr);
  } else {
    kernel!(flatBuf, numKeys, n, outRegion.ptr);
  }

  return ArrayStorage.fromWasmRegion(
    [n],
    'int32',
    outRegion,
    n,
    Int32Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
