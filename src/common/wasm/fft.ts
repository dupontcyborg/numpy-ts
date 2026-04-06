/**
 * WASM-accelerated FFT kernels.
 *
 * Provides 1D complex-to-complex FFT/IFFT and real FFT/IRFFT.
 * Uses Cooley-Tukey radix-2 for power-of-2 sizes, Bluestein's for arbitrary sizes.
 * Returns null if WASM can't handle this case.
 */

import {
  fft_c128,
  ifft_c128,
  fft_c64,
  ifft_c64,
  rfft_f64,
  irfft_f64,
  fft_scratch_size,
  fft_batch_c128,
  ifft_batch_c128,
  fft2_c128,
  ifft2_c128,
  fft2_scratch_size,
  rfft2_f64,
  rfft2_scratch_size,
  irfft2_f64,
  irfft2_scratch_size,
  rfft_batch_f64,
  irfft_batch_f64,
  rfft_batch_scratch_size,
  irfftn_3d,
  irfftn_3d_scratch_size,
} from './bins/fft.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  resolveTypedArrayPtr,
  getSharedMemory,
} from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

/**
 * WASM-accelerated 1D forward complex FFT.
 * Input: contiguous complex128 (interleaved f64) or complex64 (interleaved f32).
 * Returns null if WASM can't handle.
 */
export function wasmFft(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size; // number of complex elements
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype === 'complex128') {
    const scratchN = fft_scratch_size(size);
    const dataLen = size * 2; // interleaved re,im
    const outBytes = dataLen * 8;
    const scratchBytes = scratchN * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, dataLen, 8);
    const scratchPtr = scratchAlloc(scratchBytes);

    fft_c128(inPtr, outRegion.ptr, scratchPtr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'complex128',
      outRegion,
      dataLen,
      Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  if (dtype === 'complex64') {
    const scratchN = 4 * size + fft_scratch_size(size);
    const dataLen = size * 2;
    const outBytes = dataLen * 4;
    const scratchBytes = scratchN * 8; // scratch is always f64

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, dataLen, 4);
    const scratchPtr = scratchAlloc(scratchBytes);

    fft_c64(inPtr, outRegion.ptr, scratchPtr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'complex64',
      outRegion,
      dataLen,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return null;
}

/**
 * WASM-accelerated 1D inverse complex FFT.
 */
export function wasmIfft(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype === 'complex128') {
    const scratchN = fft_scratch_size(size);
    const dataLen = size * 2;
    const outBytes = dataLen * 8;
    const scratchBytes = scratchN * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, dataLen, 8);
    const scratchPtr = scratchAlloc(scratchBytes);

    ifft_c128(inPtr, outRegion.ptr, scratchPtr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'complex128',
      outRegion,
      dataLen,
      Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  if (dtype === 'complex64') {
    const scratchN = 4 * size + fft_scratch_size(size);
    const dataLen = size * 2;
    const outBytes = dataLen * 4;
    const scratchBytes = scratchN * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, dataLen, 4);
    const scratchPtr = scratchAlloc(scratchBytes);

    ifft_c64(inPtr, outRegion.ptr, scratchPtr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'complex64',
      outRegion,
      dataLen,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return null;
}

/**
 * WASM-accelerated 1D real-to-complex FFT.
 * Input: contiguous float64 real array. Output: complex128 with n/2+1 elements.
 */
export function wasmRfft(a: ArrayStorage, n: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.dtype !== 'float64') return null;
  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const halfN = Math.floor(n / 2) + 1;
  const scratchN = 4 * n + fft_scratch_size(n);
  const outBytes = halfN * 2 * 8;
  const scratchBytes = scratchN * 8;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, n, 8);
  const scratchPtr = scratchAlloc(scratchBytes);

  rfft_f64(inPtr, outRegion.ptr, scratchPtr, n);

  const outShape = Array.from(a.shape);
  outShape[outShape.length - 1] = halfN;
  return ArrayStorage.fromWasmRegion(
    outShape,
    'complex128',
    outRegion,
    halfN * 2,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}

/**
 * WASM-accelerated 1D complex-to-real inverse FFT.
 * Input: contiguous complex128 with n_half elements. Output: float64 with n_out elements.
 */
export function wasmIrfft(a: ArrayStorage, nOut: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.dtype !== 'complex128') return null;

  const nHalf = a.size;
  if (nOut < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const scratchN = 4 * nOut + fft_scratch_size(nOut);
  const outBytes = nOut * 8;
  const scratchBytes = scratchN * 8;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset * 2, nHalf * 2, 8);
  const scratchPtr = scratchAlloc(scratchBytes);

  irfft_f64(inPtr, outRegion.ptr, scratchPtr, nHalf, nOut);

  const outShape = Array.from(a.shape);
  outShape[outShape.length - 1] = nOut;
  return ArrayStorage.fromWasmRegion(
    outShape,
    'float64',
    outRegion,
    nOut,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}

// ============================================================
// Batch / 2D / 3D wrappers — operate on raw Float64Array buffers
// These return plain Float64Array (not ArrayStorage), so they use scratch.
// ============================================================

/**
 * WASM-accelerated 2D complex FFT or IFFT.
 * Input: contiguous complex128 2D array (rows × cols).
 */
export function wasmFft2(
  data: Float64Array,
  rows: number,
  cols: number,
  inverse: boolean
): Float64Array | null {
  const totalDataLen = rows * cols * 2;
  const scratchN = fft2_scratch_size(rows, cols);
  const totalBytes = totalDataLen * 8;
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(data.subarray(0, totalDataLen) as unknown as TypedArray);
  const outPtr = scratchAlloc(totalBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  const kernel = inverse ? ifft2_c128 : fft2_c128;
  kernel(inPtr, outPtr, scratchPtr, rows, cols);

  const mem = getSharedMemory();
  const result = new Float64Array(totalDataLen);
  new Uint8Array(result.buffer, 0, totalBytes).set(new Uint8Array(mem.buffer, outPtr, totalBytes));
  return result;
}

/**
 * WASM-accelerated batch 1D complex FFT or IFFT.
 * Processes `batch` contiguous rows of `n` complex elements each.
 */
export function wasmFftBatch(
  data: Float64Array,
  n: number,
  batch: number,
  inverse: boolean
): Float64Array | null {
  const totalDataLen = batch * n * 2;
  const scratchN = fft_scratch_size(n);
  const totalBytes = totalDataLen * 8;
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(data.subarray(0, totalDataLen) as unknown as TypedArray);
  const outPtr = scratchAlloc(totalBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  const kernel = inverse ? ifft_batch_c128 : fft_batch_c128;
  kernel(inPtr, outPtr, scratchPtr, n, batch);

  const mem = getSharedMemory();
  const result = new Float64Array(totalDataLen);
  new Uint8Array(result.buffer, 0, totalBytes).set(new Uint8Array(mem.buffer, outPtr, totalBytes));
  return result;
}

/**
 * WASM-accelerated batch real→complex rfft.
 * Input: real f64 data with `batch` rows of `n` elements.
 */
export function wasmRfftBatch(
  srcData: Float64Array,
  n: number,
  batch: number,
  inStride: number,
  outStride: number
): Float64Array | null {
  const outLen = Math.floor(n / 2) + 1;
  const outBytes = batch * outLen * 2 * 8;
  const scratchN = rfft_batch_scratch_size(n);
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(srcData.subarray(0, batch * n) as unknown as TypedArray);
  const outPtr = scratchAlloc(outBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  rfft_batch_f64(inPtr, outPtr, scratchPtr, n, batch, inStride, outStride);

  const totalOut = batch * outLen * 2;
  const mem = getSharedMemory();
  const result = new Float64Array(totalOut);
  new Uint8Array(result.buffer, 0, totalOut * 8).set(
    new Uint8Array(mem.buffer, outPtr, totalOut * 8)
  );
  return result;
}

/**
 * WASM-accelerated batch complex→real irfft.
 * Input: complex128 data with `batch` rows of `nHalf` complex elements.
 */
export function wasmIrfftBatch(
  srcData: Float64Array,
  nHalf: number,
  nOut: number,
  batch: number,
  inStride: number,
  outStride: number
): Float64Array | null {
  const outBytes = batch * nOut * 8;
  const scratchN = rfft_batch_scratch_size(nOut);
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(
    srcData.subarray(0, batch * nHalf * 2) as unknown as TypedArray
  );
  const outPtr = scratchAlloc(outBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  irfft_batch_f64(inPtr, outPtr, scratchPtr, nOut, batch, inStride, outStride);

  const totalOut = batch * nOut;
  const mem = getSharedMemory();
  const result = new Float64Array(totalOut);
  new Uint8Array(result.buffer, 0, totalOut * 8).set(
    new Uint8Array(mem.buffer, outPtr, totalOut * 8)
  );
  return result;
}

/**
 * WASM-accelerated 2D real→complex rfft2.
 * Input: real f64 (rows × cols). Output: complex128 (rows × halfCols).
 */
export function wasmRfft2(
  inputData: Float64Array | TypedArray,
  rows: number,
  cols: number
): Float64Array | null {
  const halfCols = Math.floor(cols / 2) + 1;
  const scratchN = rfft2_scratch_size(rows, cols);
  const outputLen = rows * halfCols * 2;
  const outputBytes = outputLen * 8;
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(inputData as TypedArray);
  const outPtr = scratchAlloc(outputBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  rfft2_f64(inPtr, outPtr, scratchPtr, rows, cols);

  const mem = getSharedMemory();
  const result = new Float64Array(outputLen);
  new Uint8Array(result.buffer, 0, outputBytes).set(
    new Uint8Array(mem.buffer, outPtr, outputBytes)
  );
  return result;
}

/**
 * WASM-accelerated 2D inverse real FFT (irfft2).
 * Input: complex128 (rows × colsHalf). Output: real f64 (rows × outCols).
 */
export function wasmIrfft2(
  inputData: Float64Array,
  rows: number,
  colsHalf: number,
  outCols: number
): Float64Array | null {
  const scratchN = irfft2_scratch_size(rows, outCols);
  const outputLen = rows * outCols;
  const outputBytes = outputLen * 8;
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(
    inputData.subarray(0, rows * colsHalf * 2) as unknown as TypedArray
  );
  const outPtr = scratchAlloc(outputBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  irfft2_f64(inPtr, outPtr, scratchPtr, rows, colsHalf, outCols);

  const mem = getSharedMemory();
  const result = new Float64Array(outputLen);
  new Uint8Array(result.buffer, 0, outputBytes).set(
    new Uint8Array(mem.buffer, outPtr, outputBytes)
  );
  return result;
}

/**
 * WASM-accelerated 3D inverse real FFT (irfftn for 3D).
 * Input: complex128 (d0 × d1 × d2Half). Output: real f64 (d0 × d1 × d2Out).
 */
export function wasmIrfftn3d(
  srcData: Float64Array,
  d0: number,
  d1: number,
  d2Half: number,
  d2Out: number
): Float64Array | null {
  const totalIn = d0 * d1 * d2Half * 2;
  const totalOut = d0 * d1 * d2Out;
  const outBytes = totalOut * 8;
  const scratchN = irfftn_3d_scratch_size(d0, d1, d2Out);
  const scratchBytes = scratchN * 8;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const inPtr = resolveTypedArrayPtr(srcData.subarray(0, totalIn) as unknown as TypedArray);
  const outPtr = scratchAlloc(outBytes);
  const scratchPtr = scratchAlloc(scratchBytes);

  irfftn_3d(inPtr, outPtr, scratchPtr, d0, d1, d2Half, d2Out);

  const mem = getSharedMemory();
  const result = new Float64Array(totalOut);
  new Uint8Array(result.buffer, 0, outBytes).set(new Uint8Array(mem.buffer, outPtr, outBytes));
  return result;
}
