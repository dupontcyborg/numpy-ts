/**
 * WASM-accelerated unravel_index: convert flat indices to multi-dimensional indices.
 */

import { unravel_index_i32, unravel_index_i64 } from './bins/unravel_index.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  scratchCopyIn,
  getSharedMemory,
} from './runtime';
import type { WasmRegion } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

/**
 * WASM-accelerated unravel_index for flat indices with C-order strides.
 * Returns ndim ArrayStorage arrays, or null if WASM can't handle this case.
 */
export function wasmUnravelIndex(indices: ArrayStorage, shape: number[]): ArrayStorage[] | null {
  if (!indices.isCContiguous) return null;

  const N = indices.size;
  const ndim = shape.length;
  if (ndim === 0 || N === 0) return null;
  if (N < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const bpe = 4; // i32

  // Allocate ndim separate persistent output regions (one per dimension)
  const outRegions: WasmRegion[] = [];
  for (let d = 0; d < ndim; d++) {
    const region = wasmMalloc(N * bpe);
    if (!region) {
      for (const r of outRegions) r.release();
      return null;
    }
    outRegions.push(region);
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const idxOff = indices.offset;
  const idxData = indices.data;
  const dtype = indices.dtype;
  const use64 = dtype === 'int64' || dtype === 'uint64';

  // i64/u64 path: use i64 kernel directly (8 bytes per element)
  if (use64) {
    const bpe64 = 8;
    // Re-allocate output regions for i64 (previous ones were i32-sized)
    for (const r of outRegions) r.release();
    outRegions.length = 0;
    for (let d = 0; d < ndim; d++) {
      const region = wasmMalloc(N * bpe64);
      if (!region) {
        for (const r of outRegions) r.release();
        return null;
      }
      outRegions.push(region);
    }

    const indicesPtr = resolveInputPtr(
      idxData,
      indices.isWasmBacked,
      indices.wasmPtr,
      idxOff,
      N,
      bpe64
    );

    const strides64 = new BigInt64Array(ndim);
    let stride64 = 1n;
    for (let i = ndim - 1; i >= 0; i--) {
      strides64[i] = stride64;
      stride64 *= BigInt(shape[i]!);
    }
    const stridesPtr = scratchCopyIn(strides64 as unknown as TypedArray);
    const shape64 = new BigInt64Array(shape.map(BigInt));
    const shapePtr = scratchCopyIn(shape64 as unknown as TypedArray);

    const outScratchPtr = scratchAlloc(ndim * N * bpe64);
    unravel_index_i64(indicesPtr, outScratchPtr, N, stridesPtr, shapePtr, ndim);

    const mem = getSharedMemory();
    for (let d = 0; d < ndim; d++) {
      const src = new BigInt64Array(mem.buffer, outScratchPtr + d * N * bpe64, N);
      const dst = new BigInt64Array(mem.buffer, outRegions[d]!.ptr, N);
      dst.set(src);
    }

    const outputShape = Array.from(indices.shape);
    const results: ArrayStorage[] = [];
    for (let d = 0; d < ndim; d++) {
      results.push(
        ArrayStorage.fromWasmRegion(
          outputShape,
          'int64',
          outRegions[d]!,
          N,
          BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
        )
      );
    }
    return results;
  }

  // i32/u32 path: use i32 kernel (u32 has same bit pattern for non-negative values)
  let indicesPtr: number;
  if (dtype === 'int32' || dtype === 'uint32') {
    indicesPtr = resolveInputPtr(idxData, indices.isWasmBacked, indices.wasmPtr, idxOff, N, bpe);
  } else {
    const i32 = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      i32[i] = Number(idxData[idxOff + i]);
    }
    indicesPtr = scratchCopyIn(i32 as unknown as TypedArray);
  }

  const strides = new Int32Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  const stridesPtr = scratchCopyIn(strides as unknown as TypedArray);
  const shapeI32 = new Int32Array(shape);
  const shapePtr = scratchCopyIn(shapeI32 as unknown as TypedArray);

  const outScratchPtr = scratchAlloc(ndim * N * bpe);
  unravel_index_i32(indicesPtr, outScratchPtr, N, stridesPtr, shapePtr, ndim);

  const mem = getSharedMemory();
  for (let d = 0; d < ndim; d++) {
    const src = new Int32Array(mem.buffer, outScratchPtr + d * N * bpe, N);
    const dst = new Int32Array(mem.buffer, outRegions[d]!.ptr, N);
    dst.set(src);
  }

  const outputShape = Array.from(indices.shape);
  const results: ArrayStorage[] = [];
  for (let d = 0; d < ndim; d++) {
    results.push(
      ArrayStorage.fromWasmRegion(
        outputShape,
        'int32',
        outRegions[d]!,
        N,
        Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      )
    );
  }

  return results;
}
