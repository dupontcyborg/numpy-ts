/**
 * WASM-accelerated unravel_index: convert flat indices to multi-dimensional indices.
 * Uses native i32/i64 kernels for integer inputs (no JS conversion overhead),
 * with f64 kernel as fallback. All kernels output f64 directly.
 */

import {
  unravel_index_i32_f64,
  unravel_index_i64_f64,
  unravel_index_f64,
} from './bins/unravel_index.wasm';
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

const BASE_THRESHOLD = 32;

/**
 * WASM-accelerated unravel_index for flat indices with C-order strides.
 * Returns ndim ArrayStorage arrays (float64), or null if WASM can't handle this case.
 */
export function wasmUnravelIndex(indices: ArrayStorage, shape: number[]): ArrayStorage[] | null {
  if (!indices.isCContiguous) return null;

  const N = indices.size;
  const ndim = shape.length;
  if (ndim === 0 || N === 0) return null;
  if (N < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = indices.dtype;
  const idxData = indices.data;
  const idxOff = indices.offset;
  const f64Bpe = 8;

  // Allocate f64 output regions
  const outRegions: WasmRegion[] = [];
  for (let d = 0; d < ndim; d++) {
    const region = wasmMalloc(N * f64Bpe);
    if (!region) {
      for (const r of outRegions) r.release();
      return null;
    }
    outRegions.push(region);
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // All kernels output f64 directly into scratch, then copy to persistent regions
  const outScratchPtr = scratchAlloc(ndim * N * f64Bpe);

  if (dtype === 'int64' || dtype === 'uint64') {
    // i64 kernel: native BigInt input, f64 output
    const bpe64 = 8;
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

    unravel_index_i64_f64(indicesPtr, outScratchPtr, N, stridesPtr, shapePtr, ndim);
  } else if (dtype === 'int32' || dtype === 'uint32') {
    // i32 kernel: native int input, f64 output
    const bpe32 = 4;
    const indicesPtr = resolveInputPtr(
      idxData,
      indices.isWasmBacked,
      indices.wasmPtr,
      idxOff,
      N,
      bpe32
    );

    const strides = new Int32Array(ndim);
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i]!;
    }
    const stridesPtr = scratchCopyIn(strides as unknown as TypedArray);
    const shapeI32 = new Int32Array(shape);
    const shapePtr = scratchCopyIn(shapeI32 as unknown as TypedArray);

    unravel_index_i32_f64(indicesPtr, outScratchPtr, N, stridesPtr, shapePtr, ndim);
  } else {
    // f64 fallback: convert input to f64 if needed
    let indicesPtr: number;
    if (dtype === 'float64') {
      indicesPtr = resolveInputPtr(
        idxData,
        indices.isWasmBacked,
        indices.wasmPtr,
        idxOff,
        N,
        f64Bpe
      );
    } else {
      const f64 = new Float64Array(N);
      for (let i = 0; i < N; i++) f64[i] = Number(idxData[idxOff + i]);
      indicesPtr = scratchCopyIn(f64 as unknown as TypedArray);
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

    unravel_index_f64(indicesPtr, outScratchPtr, N, stridesPtr, shapePtr, ndim);
  }

  // Copy f64 scratch output to persistent regions
  const mem = getSharedMemory();
  for (let d = 0; d < ndim; d++) {
    const src = new Float64Array(mem.buffer, outScratchPtr + d * N * f64Bpe, N);
    const dst = new Float64Array(mem.buffer, outRegions[d]!.ptr, N);
    dst.set(src);
  }

  const outputShape = Array.from(indices.shape);
  const results: ArrayStorage[] = [];
  for (let d = 0; d < ndim; d++) {
    results.push(
      ArrayStorage.fromWasmRegion(
        outputShape,
        'float64',
        outRegions[d]!,
        N,
        Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      )
    );
  }

  return results;
}
