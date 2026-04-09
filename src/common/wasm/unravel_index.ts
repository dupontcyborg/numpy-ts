/**
 * WASM-accelerated unravel_index: convert flat indices to multi-dimensional indices.
 */

import { unravel_index_f64 } from './bins/unravel_index.wasm';
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

  // Allocate ndim separate persistent output regions (f64, 8 bytes per element)
  const outBpe = 8;
  const outRegions: WasmRegion[] = [];
  for (let d = 0; d < ndim; d++) {
    const region = wasmMalloc(N * outBpe);
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

  // Always output float64 (our convention for index-returning ops)
  const f64Bpe = 8;
  let indicesPtr: number;
  if (dtype === 'float64') {
    indicesPtr = resolveInputPtr(idxData, indices.isWasmBacked, indices.wasmPtr, idxOff, N, f64Bpe);
  } else {
    // Convert any non-f64 input (int32, uint32, float32, etc.) to f64 in scratch
    const f64 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      f64[i] = Number(idxData[idxOff + i]);
    }
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

  // Kernel writes f64 output directly into the persistent output regions
  const outScratchPtr = scratchAlloc(ndim * N * f64Bpe);
  unravel_index_f64(indicesPtr, outScratchPtr, N, stridesPtr, shapePtr, ndim);

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
