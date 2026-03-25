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
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
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
    const inBytes = dataLen * 8;
    const outBytes = dataLen * 8;
    const scratchBytes = scratchN * 8;

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    fft_c128(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex128');
  }

  if (dtype === 'complex64') {
    const scratchN = 4 * size + fft_scratch_size(size);
    const dataLen = size * 2;
    const inBytes = dataLen * 4; // f32
    const outBytes = dataLen * 4;
    const scratchBytes = scratchN * 8; // scratch is always f64

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    fft_c64(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex64');
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
    const inBytes = dataLen * 8;
    const outBytes = dataLen * 8;
    const scratchBytes = scratchN * 8;

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    ifft_c128(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex128');
  }

  if (dtype === 'complex64') {
    const scratchN = 4 * size + fft_scratch_size(size);
    const dataLen = size * 2;
    const inBytes = dataLen * 4;
    const outBytes = dataLen * 4;
    const scratchBytes = scratchN * 8;

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    ifft_c64(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex64');
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
  // scratch: 2*n (pack to complex) + 2*n (bluestein output) + fft_scratch_size(n)
  const scratchN = 4 * n + fft_scratch_size(n);
  const inBytes = n * 8;
  const outBytes = halfN * 2 * 8;
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const aData = a.data.subarray(a.offset, a.offset + n) as TypedArray;
  const inPtr = copyIn(aData);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  rfft_f64(inPtr, outPtr, scratchPtr, n);

  const outData = copyOut(
    outPtr,
    halfN * 2,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  const outShape = Array.from(a.shape);
  outShape[outShape.length - 1] = halfN;
  return ArrayStorage.fromData(outData, outShape, 'complex128');
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

  // scratch: 2*nOut (full spectrum) + 2*nOut (bluestein output) + fft_scratch_size(nOut)
  const scratchN = 4 * nOut + fft_scratch_size(nOut);
  const inBytes = nHalf * 2 * 8;
  const outBytes = nOut * 8;
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const aData = a.data.subarray(a.offset * 2, (a.offset + nHalf) * 2) as TypedArray;
  const inPtr = copyIn(aData);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  irfft_f64(inPtr, outPtr, scratchPtr, nHalf, nOut);

  const outData = copyOut(
    outPtr,
    nOut,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  const outShape = Array.from(a.shape);
  outShape[outShape.length - 1] = nOut;
  return ArrayStorage.fromData(outData, outShape, 'float64');
}

// ============================================================
// Batch / 2D / 3D wrappers
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

  ensureMemory(totalBytes * 2 + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(data.subarray(0, totalDataLen) as unknown as TypedArray);
  const outPtr = alloc(totalBytes);
  const scratchPtr = alloc(scratchBytes);

  const kernel = inverse ? ifft2_c128 : fft2_c128;
  kernel(inPtr, outPtr, scratchPtr, rows, cols);

  return copyOut(
    outPtr,
    totalDataLen,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
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

  ensureMemory(totalBytes * 2 + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(data.subarray(0, totalDataLen) as unknown as TypedArray);
  const outPtr = alloc(totalBytes);
  const scratchPtr = alloc(scratchBytes);

  const kernel = inverse ? ifft_batch_c128 : fft_batch_c128;
  kernel(inPtr, outPtr, scratchPtr, n, batch);

  return copyOut(
    outPtr,
    totalDataLen,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
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
  const inBytes = batch * n * 8;
  const outBytes = batch * outLen * 2 * 8;
  const scratchN = rfft_batch_scratch_size(n);
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(srcData.subarray(0, batch * n) as unknown as TypedArray);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  rfft_batch_f64(inPtr, outPtr, scratchPtr, n, batch, inStride, outStride);

  return copyOut(
    outPtr,
    batch * outLen * 2,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
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
  const inBytes = batch * nHalf * 2 * 8;
  const outBytes = batch * nOut * 8;
  const scratchN = rfft_batch_scratch_size(nOut);
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(srcData.subarray(0, batch * nHalf * 2) as unknown as TypedArray);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  irfft_batch_f64(inPtr, outPtr, scratchPtr, nOut, batch, inStride, outStride);

  return copyOut(
    outPtr,
    batch * nOut,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
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
  const inputBytes = rows * cols * 8;
  const outputBytes = rows * halfCols * 2 * 8;
  const scratchBytes = scratchN * 8;

  ensureMemory(inputBytes + outputBytes + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(inputData as TypedArray);
  const outPtr = alloc(outputBytes);
  const scratchPtr = alloc(scratchBytes);

  rfft2_f64(inPtr, outPtr, scratchPtr, rows, cols);

  return copyOut(
    outPtr,
    rows * halfCols * 2,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
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
  const inputBytes = rows * colsHalf * 2 * 8;
  const outputBytes = rows * outCols * 8;
  const scratchBytes = scratchN * 8;

  ensureMemory(inputBytes + outputBytes + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(inputData.subarray(0, rows * colsHalf * 2) as unknown as TypedArray);
  const outPtr = alloc(outputBytes);
  const scratchPtr = alloc(scratchBytes);

  irfft2_f64(inPtr, outPtr, scratchPtr, rows, colsHalf, outCols);

  return copyOut(
    outPtr,
    rows * outCols,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
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

  const inBytes = totalIn * 8;
  const outBytes = totalOut * 8;
  const scratchN = irfftn_3d_scratch_size(d0, d1, d2Out);
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const inPtr = copyIn(srcData.subarray(0, totalIn) as unknown as TypedArray);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  irfftn_3d(inPtr, outPtr, scratchPtr, d0, d1, d2Half, d2Out);

  return copyOut(
    outPtr,
    totalOut,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => Float64Array
  ) as unknown as Float64Array;
}
