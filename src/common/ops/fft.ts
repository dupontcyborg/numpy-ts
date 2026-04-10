/**
 * FFT (Fast Fourier Transform) operations
 *
 * Implements NumPy's np.fft module:
 * - fft, ifft: 1D FFT
 * - fft2, ifft2: 2D FFT
 * - fftn, ifftn: N-dimensional FFT
 * - rfft, irfft: Real FFT
 * - rfft2, irfft2: 2D Real FFT
 * - rfftn, irfftn: N-dimensional Real FFT
 * - hfft, ihfft: Hermitian FFT
 * - fftfreq, rfftfreq: FFT frequencies
 * - fftshift, ifftshift: Shift zero-frequency to center
 *
 * @module ops/fft
 */

import { ArrayStorage } from '../storage';
import { Complex } from '../complex';
import { isComplexDType, throwIfComplex, fftResultDtype, hasFloat16, type DType } from '../dtype';
import { roll as shapeRoll } from './shape';
import {
  wasmFft,
  wasmIfft,
  wasmRfft,
  wasmIrfft,
  wasmFft2,
  wasmFftBatch,
  wasmRfftBatch,
  wasmIrfftBatch,
  wasmRfft2,
  wasmIrfft2,
  wasmIrfftn3d,
} from '../wasm/fft';
import { wasmConfig } from '../wasm/config';

/**
 * Cooley-Tukey FFT algorithm (radix-2)
 * For non-power-of-2 sizes, we use Bluestein's algorithm
 */
function fftCore(real: Float64Array, imag: Float64Array, inverse: boolean): void {
  const n = real.length;

  if (n === 0) return;
  if (n === 1) return;

  // Check if n is a power of 2
  if ((n & (n - 1)) === 0) {
    // Use Cooley-Tukey for power of 2
    cooleyTukeyFFT(real, imag, inverse);
  } else {
    // Use Bluestein's algorithm for arbitrary sizes
    bluesteinFFT(real, imag, inverse);
  }
}

// Cache for precomputed twiddle factors
const twiddleCache = new Map<string, { cos: Float64Array; sin: Float64Array }>();

/**
 * Get or compute twiddle factors for a given size
 */
function getTwiddleFactors(n: number, inverse: boolean): { cos: Float64Array; sin: Float64Array } {
  const key = `${n}_${inverse}`;
  let cached = twiddleCache.get(key);
  if (cached) return cached;

  // Precompute all twiddle factors for this size
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  const sign = inverse ? 1 : -1;

  for (let k = 0; k < n / 2; k++) {
    const angle = (sign * 2 * Math.PI * k) / n;
    cos[k] = Math.cos(angle);
    sin[k] = Math.sin(angle);
  }

  cached = { cos, sin };
  twiddleCache.set(key, cached);

  // Limit cache size
  if (twiddleCache.size > 100) {
    const firstKey = twiddleCache.keys().next().value as string;
    twiddleCache.delete(firstKey);
  }

  return cached;
}

/**
 * Cooley-Tukey radix-2 decimation-in-time FFT
 * Optimized with precomputed twiddle factors
 */
function cooleyTukeyFFT(real: Float64Array, imag: Float64Array, inverse: boolean): void {
  const n = real.length;

  // Bit reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      // Swap
      let temp = real[i]!;
      real[i] = real[j]!;
      real[j] = temp;
      temp = imag[i]!;
      imag[i] = imag[j]!;
      imag[j] = temp;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey iterative FFT with precomputed twiddle factors
  const { cos: twCos, sin: twSin } = getTwiddleFactors(n, inverse);

  for (let size = 2; size <= n; size *= 2) {
    const halfsize = size >> 1;
    const tablestep = n / size;

    for (let i = 0; i < n; i += size) {
      for (let k = 0, twidx = 0; k < halfsize; k++, twidx += tablestep) {
        const cos = twCos[twidx]!;
        const sin = twSin[twidx]!;

        const evenIdx = i + k;
        const oddIdx = i + k + halfsize;

        const evenReal = real[evenIdx]!;
        const evenImag = imag[evenIdx]!;
        const oddReal = real[oddIdx]!;
        const oddImag = imag[oddIdx]!;

        // Twiddle factor multiplication
        const tReal = cos * oddReal - sin * oddImag;
        const tImag = cos * oddImag + sin * oddReal;

        // Butterfly
        real[evenIdx] = evenReal + tReal;
        imag[evenIdx] = evenImag + tImag;
        real[oddIdx] = evenReal - tReal;
        imag[oddIdx] = evenImag - tImag;
      }
    }
  }

  // Scale for inverse FFT
  if (inverse) {
    const scale = 1 / n;
    for (let i = 0; i < n; i++) {
      real[i] = real[i]! * scale;
      imag[i] = imag[i]! * scale;
    }
  }
}

/**
 * Bluestein's FFT algorithm for arbitrary-length FFTs
 * Uses chirp-z transform approach
 *
 * The DFT: X[k] = sum_{j=0}^{n-1} x[j] * exp(-2*pi*i*j*k/n)
 *
 * Using identity 2jk = j² + k² - (k-j)²:
 * X[k] = w[k] * sum_{j=0}^{n-1} [x[j] * w[j]] * conj(w[k-j])
 *
 * where w[k] = exp(-pi*i*k²/n)
 *
 * This is a convolution that can be computed via FFT.
 */
function bluesteinFFT(real: Float64Array, imag: Float64Array, inverse: boolean): void {
  const n = real.length;

  // Find next power of 2 >= 2n - 1 for the convolution
  let m = 1;
  while (m < 2 * n - 1) {
    m *= 2;
  }

  // For forward DFT: w[k] = exp(-pi*i*k²/n)
  // For inverse DFT: w[k] = exp(+pi*i*k²/n)
  const sign = inverse ? 1 : -1;

  // Precompute chirp: w[k] = exp(sign * pi * i * k² / n)
  const wReal = new Float64Array(n);
  const wImag = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const angle = (sign * Math.PI * k * k) / n;
    wReal[k] = Math.cos(angle);
    wImag[k] = Math.sin(angle);
  }

  // Sequence a: input multiplied by w (padded to length m)
  // a[j] = x[j] * w[j]
  const aReal = new Float64Array(m);
  const aImag = new Float64Array(m);
  for (let j = 0; j < n; j++) {
    const wr = wReal[j]!;
    const wi = wImag[j]!;
    // (xr + i*xi) * (wr + i*wi) = (xr*wr - xi*wi) + i*(xi*wr + xr*wi)
    aReal[j] = real[j]! * wr - imag[j]! * wi;
    aImag[j] = imag[j]! * wr + real[j]! * wi;
  }

  // Sequence b: conj(w) with wrap-around for negative indices
  // b[k] = conj(w[k]) for k = 0..n-1
  // b[m-k] = conj(w[k]) for k = 1..n-1 (represents b[-k])
  const bReal = new Float64Array(m);
  const bImag = new Float64Array(m);
  bReal[0] = wReal[0]!;
  bImag[0] = -wImag[0]!; // conj
  for (let k = 1; k < n; k++) {
    bReal[k] = wReal[k]!;
    bImag[k] = -wImag[k]!; // conj
    bReal[m - k] = wReal[k]!;
    bImag[m - k] = -wImag[k]!; // conj
  }

  // FFT of both sequences
  cooleyTukeyFFT(aReal, aImag, false);
  cooleyTukeyFFT(bReal, bImag, false);

  // Pointwise multiplication in frequency domain
  for (let k = 0; k < m; k++) {
    const ar = aReal[k]!;
    const ai = aImag[k]!;
    const br = bReal[k]!;
    const bi = bImag[k]!;
    aReal[k] = ar * br - ai * bi;
    aImag[k] = ar * bi + ai * br;
  }

  // Inverse FFT to get convolution result
  // IFFT(DFT(a) * DFT(b)) = circular convolution of a and b
  cooleyTukeyFFT(aReal, aImag, true);

  // Multiply by w to complete Bluestein: X[k] = w[k] * convolution[k]
  for (let k = 0; k < n; k++) {
    const cr = aReal[k]!;
    const ci = aImag[k]!;
    const wr = wReal[k]!;
    const wi = wImag[k]!;
    // (cr + i*ci) * (wr + i*wi) = (cr*wr - ci*wi) + i*(ci*wr + cr*wi)
    real[k] = cr * wr - ci * wi;
    imag[k] = ci * wr + cr * wi;
  }

  // Scale for inverse DFT
  if (inverse) {
    for (let k = 0; k < n; k++) {
      real[k] = real[k]! / n;
      imag[k] = imag[k]! / n;
    }
  }
}

/**
 * Compute the 1-D discrete Fourier Transform.
 *
 * @param a - Input array storage
 * @param n - Length of the transformed axis (default: length of a)
 * @param axis - Axis along which to compute the FFT (default: -1, last axis)
 * @param norm - Normalization mode: 'backward', 'ortho', or 'forward' (default: 'backward')
 * @returns Complex array containing the FFT result
 */
export function fft(
  a: ArrayStorage,
  n?: number,
  axis: number = -1,
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  // WASM fast path: 1D FFT on last axis, no truncation/padding, backward norm
  if (
    n === undefined &&
    (axis === -1 || axis === a.ndim - 1) &&
    norm === 'backward' &&
    a.ndim === 1
  ) {
    const isAlreadyComplex = isComplexDType(a.dtype);
    const complexA = isAlreadyComplex ? a : toComplex(a);
    try {
      const result = wasmFft(complexA);
      if (result) return result;
    } finally {
      if (!isAlreadyComplex) complexA.dispose();
    }
  }
  // Always pass explicit axis for 1D FFT
  return fftnd(a, n !== undefined ? [n] : undefined, [axis], norm, false);
}

/**
 * Compute the 1-D inverse discrete Fourier Transform.
 *
 * @param a - Input array storage
 * @param n - Length of the transformed axis (default: length of a)
 * @param axis - Axis along which to compute the IFFT (default: -1, last axis)
 * @param norm - Normalization mode: 'backward', 'ortho', or 'forward' (default: 'backward')
 * @returns Complex array containing the IFFT result
 */
export function ifft(
  a: ArrayStorage,
  n?: number,
  axis: number = -1,
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  // WASM fast path: 1D IFFT on last axis, no truncation/padding, backward norm
  if (
    n === undefined &&
    (axis === -1 || axis === a.ndim - 1) &&
    norm === 'backward' &&
    a.ndim === 1
  ) {
    const isAlreadyComplex = isComplexDType(a.dtype);
    const complexA = isAlreadyComplex ? a : toComplex(a);
    try {
      const result = wasmIfft(complexA);
      if (result) return result;
    } finally {
      if (!isAlreadyComplex) complexA.dispose();
    }
  }
  // Always pass explicit axis for 1D IFFT
  return fftnd(a, n !== undefined ? [n] : undefined, [axis], norm, true);
}

/**
 * Compute the 2-D discrete Fourier Transform.
 *
 * @param a - Input array storage
 * @param s - Shape of the output (default: shape of input along axes)
 * @param axes - Axes along which to compute the FFT (default: [-2, -1])
 * @param norm - Normalization mode
 * @returns Complex array containing the FFT result
 */
export function fft2(
  a: ArrayStorage,
  s?: [number, number],
  axes: [number, number] = [-2, -1],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  return fftnd(a, s, axes, norm, false);
}

/**
 * Compute the 2-D inverse discrete Fourier Transform.
 *
 * @param a - Input array storage
 * @param s - Shape of the output
 * @param axes - Axes along which to compute the IFFT
 * @param norm - Normalization mode
 * @returns Complex array containing the IFFT result
 */
export function ifft2(
  a: ArrayStorage,
  s?: [number, number],
  axes: [number, number] = [-2, -1],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  return fftnd(a, s, axes, norm, true);
}

/**
 * Compute the N-D discrete Fourier Transform.
 *
 * @param a - Input array storage
 * @param s - Shape of the output along transform axes
 * @param axes - Axes along which to compute the FFT
 * @param norm - Normalization mode
 * @returns Complex array containing the FFT result
 */
export function fftn(
  a: ArrayStorage,
  s?: number[],
  axes?: number[],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  return fftnd(a, s, axes, norm, false);
}

/**
 * Compute the N-D inverse discrete Fourier Transform.
 *
 * @param a - Input array storage
 * @param s - Shape of the output along transform axes
 * @param axes - Axes along which to compute the IFFT
 * @param norm - Normalization mode
 * @returns Complex array containing the IFFT result
 */
export function ifftn(
  a: ArrayStorage,
  s?: number[],
  axes?: number[],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  return fftnd(a, s, axes, norm, true);
}

/**
 * Internal N-D FFT implementation
 */
function fftnd(
  a: ArrayStorage,
  s?: number[],
  axes?: number[],
  norm: 'backward' | 'ortho' | 'forward' = 'backward',
  inverse: boolean = false
): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const cDtype = fftResultDtype(a.dtype as DType);

  // Handle empty arrays
  if (a.size === 0) {
    return ArrayStorage.zeros(shape, cDtype);
  }

  // Handle 0-D arrays
  if (ndim === 0) {
    const result = ArrayStorage.zeros([1], cDtype);
    const val = a.iget(0);
    const re = val instanceof Complex ? val.re : Number(val);
    const im = val instanceof Complex ? val.im : 0;
    const data = result.data as Float64Array | Float32Array;
    data[0] = re;
    data[1] = im;
    // Reshape to scalar
    return ArrayStorage.fromData(result.data, [], cDtype);
  }

  // Normalize axes
  let axesList: number[];
  if (axes === undefined) {
    // Default: all axes for fftn (when called with axes=undefined)
    if (s === undefined) {
      // No shape specified: use all axes
      axesList = Array.from({ length: ndim }, (_, i) => i);
    } else {
      axesList = [];
      for (let i = 0; i < s.length; i++) {
        axesList.push(ndim - s.length + i);
      }
    }
  } else {
    axesList = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  // Determine output shape
  const outShape = [...shape];
  if (s !== undefined) {
    for (let i = 0; i < s.length; i++) {
      const ax = axesList[i]!;
      outShape[ax] = s[i]!;
    }
  }

  // Convert input to complex
  let result = toComplex(a);

  // Pad or truncate if needed
  if (s !== undefined) {
    const prev = result;
    result = padOrTruncate(result, outShape, axesList);
    if (result !== prev) prev.dispose();
  }

  // WASM fast path for 2D FFT: single call handles both axes (row FFT + transpose + col FFT)
  if (axesList.length === 2 && result.ndim === 2 && norm === 'backward' && result.isCContiguous) {
    const rows = result.shape[axesList[0]!]!;
    const cols = result.shape[axesList[1]!]!;
    const isStandardAxes = axesList[0] === 0 && axesList[1] === 1;

    if (
      isStandardAxes &&
      rows >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier &&
      cols >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier
    ) {
      const srcData = result.data as Float64Array | Float32Array;
      const outData = wasmFft2(srcData, rows, cols, inverse);
      if (outData) {
        result.dispose();
        return ArrayStorage.fromData(outData, [rows, cols], cDtype);
      }
    }
  }

  // Apply FFT along each axis
  for (const axis of axesList) {
    const prev = result;
    result = fft1dAlongAxis(result, axis, inverse, norm);
    prev.dispose();
  }

  return result;
}

/**
 * Convert array to complex128
 * Optimized with direct typed array access
 */
function toComplex(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype as DType;
  const shape = Array.from(a.shape);
  const size = a.size;
  const outDtype = fftResultDtype(dtype);

  if (dtype === 'complex128' || dtype === 'complex64') {
    // Already complex — promote to target complex dtype
    const result = ArrayStorage.zeros(shape, outDtype);
    const resultData = result.data as Float64Array | Float32Array;
    const srcData = a.data as Float64Array | Float32Array;

    for (let i = 0; i < size * 2; i++) {
      resultData[i] = srcData[i]!;
    }
    return result;
  }

  // Real array: convert to complex using direct typed array access
  const result = ArrayStorage.zeros(shape, outDtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data;

  // Direct typed array access based on dtype
  if (dtype === 'float64') {
    const typed = srcData as Float64Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
      // imag part already 0 from zeros()
    }
  } else if (dtype === 'float32') {
    const typed = srcData as Float32Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'int32') {
    const typed = srcData as Int32Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'int16') {
    const typed = srcData as Int16Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'int8') {
    const typed = srcData as Int8Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'uint32') {
    const typed = srcData as Uint32Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'uint16') {
    const typed = srcData as Uint16Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'uint8') {
    const typed = srcData as Uint8Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = typed[i]!;
    }
  } else if (dtype === 'int64' || dtype === 'uint64') {
    const typed = srcData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = Number(typed[i]!);
    }
  } else {
    // Fallback for bool and other types
    for (let i = 0; i < size; i++) {
      resultData[i * 2] = Number(srcData[i]!);
    }
  }

  return result;
}

/**
 * Pad or truncate array along specified axes
 */
function padOrTruncate(a: ArrayStorage, targetShape: number[], axes: number[]): ArrayStorage {
  const currentShape = Array.from(a.shape);
  let result = a;

  for (const axis of axes) {
    const currentSize = currentShape[axis]!;
    const targetSize = targetShape[axis]!;

    if (currentSize === targetSize) continue;

    if (targetSize > currentSize) {
      // Pad with zeros
      result = padAxis(result, axis, targetSize);
    } else {
      // Truncate
      result = truncateAxis(result, axis, targetSize);
    }
    currentShape[axis] = targetSize;
  }

  return result;
}

/**
 * Pad array along a single axis
 */
function padAxis(a: ArrayStorage, axis: number, newSize: number): ArrayStorage {
  const shape = Array.from(a.shape);
  const oldSize = shape[axis]!;
  shape[axis] = newSize;

  const result = ArrayStorage.zeros(shape, a.dtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data as Float64Array | Float32Array;

  // Calculate strides for iteration
  const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1);
  const innerSize = shape.slice(axis + 1).reduce((a, b) => a * b, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let k = 0; k < oldSize; k++) {
      for (let inner = 0; inner < innerSize; inner++) {
        const srcIdx = (outer * oldSize + k) * innerSize + inner;
        const dstIdx = (outer * newSize + k) * innerSize + inner;
        resultData[dstIdx * 2] = srcData[srcIdx * 2]!;
        resultData[dstIdx * 2 + 1] = srcData[srcIdx * 2 + 1]!;
      }
    }
  }

  return result;
}

/**
 * Truncate array along a single axis
 */
function truncateAxis(a: ArrayStorage, axis: number, newSize: number): ArrayStorage {
  const shape = Array.from(a.shape);
  const oldSize = shape[axis]!;
  shape[axis] = newSize;

  const result = ArrayStorage.zeros(shape, a.dtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data as Float64Array;

  const outerSize = shape.slice(0, axis).reduce((a, b) => a * b, 1);
  const innerSize = shape.slice(axis + 1).reduce((a, b) => a * b, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let k = 0; k < newSize; k++) {
      for (let inner = 0; inner < innerSize; inner++) {
        const srcIdx = (outer * oldSize + k) * innerSize + inner;
        const dstIdx = (outer * newSize + k) * innerSize + inner;
        resultData[dstIdx * 2] = srcData[srcIdx * 2]!;
        resultData[dstIdx * 2 + 1] = srcData[srcIdx * 2 + 1]!;
      }
    }
  }

  return result;
}

/**
 * Apply 1D FFT along a specific axis
 */
const FFT_WASM_THRESHOLD = 32;

/**
 * WASM batch rfft: processes all last-axis rfft's in a single WASM call.
 * Input: real array, C-contiguous. Transforms along last axis.
 * Returns null if WASM can't handle.
 */
function wasmBatchRfft(a: ArrayStorage, n: number): ArrayStorage | null {
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const outLen = Math.floor(n / 2) + 1;
  const batch = shape.slice(0, ndim - 1).reduce((acc, s) => acc * s, 1);
  const inStride = n;
  const outStride = outLen * 2;

  let srcData: Float64Array | Float32Array;
  if (a.dtype === 'float64') {
    srcData = a.data as Float64Array;
  } else if (a.dtype === 'float32') {
    srcData = a.data as Float32Array;
  } else if (a.dtype === 'int64' || a.dtype === 'uint64') {
    const bigData = a.data as BigInt64Array | BigUint64Array;
    srcData = new Float64Array(a.size);
    for (let i = 0; i < a.size; i++) srcData[i] = Number(bigData[a.offset + i]!);
  } else {
    srcData = Float64Array.from(
      a.data.subarray(a.offset, a.offset + a.size) as unknown as ArrayLike<number>
    );
  }

  const outData = wasmRfftBatch(srcData, n, batch, inStride, outStride);
  if (!outData) return null;

  const outShape = [...shape];
  outShape[ndim - 1] = outLen;
  return ArrayStorage.fromData(outData, outShape, fftResultDtype(a.dtype as DType));
}

/**
 * WASM batch irfft: processes all last-axis irfft's in a single WASM call.
 * Input: complex array, C-contiguous. Transforms along last axis.
 * Returns null if WASM can't handle.
 */
function wasmBatchIrfft(a: ArrayStorage, nOut: number): ArrayStorage | null {
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const nHalf = shape[ndim - 1]!;
  const batch = shape.slice(0, ndim - 1).reduce((acc, s) => acc * s, 1);
  const inStride = nHalf * 2;
  const outStride = nOut;

  const cplx = toComplex(a);
  const isC64 = cplx.dtype === 'complex64';
  const outDtype: DType = isC64 ? 'float32' : 'float64';

  // complex64 WASM-backed data may be Float64Array view — extract to proper Float32Array
  let srcData: Float64Array | Float32Array;
  if (isC64 && !(cplx.data instanceof Float32Array)) {
    const len = cplx.size * 2;
    srcData = new Float32Array(len);
    const src = cplx.data;
    for (let i = 0; i < len; i++) srcData[i] = Number(src[i]!);
  } else {
    srcData = cplx.data as Float64Array | Float32Array;
  }

  const outData = wasmIrfftBatch(srcData, nHalf, nOut, batch, inStride, outStride);
  if (!outData) return null;

  const outShape = [...shape];
  outShape[ndim - 1] = nOut;
  return ArrayStorage.fromData(outData, outShape, outDtype);
}

function fft1dAlongAxis(
  a: ArrayStorage,
  axis: number,
  inverse: boolean,
  norm: 'backward' | 'ortho' | 'forward'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const n = shape[axis]!;

  if (n === 0) return a;

  const result = ArrayStorage.zeros(shape, a.dtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data as Float64Array | Float32Array;

  // Calculate dimensions
  const outerSize = shape.slice(0, axis).reduce((acc, s) => acc * s, 1);
  const innerSize = shape.slice(axis + 1).reduce((acc, s) => acc * s, 1);

  // WASM fast path: when innerSize === 1, rows are contiguous interleaved complex data.
  // Batch all rows into a single WASM call to avoid per-row copy overhead.
  if (innerSize === 1 && n >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier) {
    const totalDataLen = outerSize * n * 2;
    const outData = wasmFftBatch(srcData, n, outerSize, inverse);
    if (outData) {
      resultData.set(outData);

      // Apply normalization adjustments for non-backward modes
      if (norm === 'ortho') {
        const scale = inverse ? Math.sqrt(n) : 1 / Math.sqrt(n);
        for (let i = 0; i < totalDataLen; i++) resultData[i] = resultData[i]! * scale;
      } else if (norm === 'forward' && !inverse) {
        const scale = 1 / n;
        for (let i = 0; i < totalDataLen; i++) resultData[i] = resultData[i]! * scale;
      } else if (norm === 'forward' && inverse) {
        // WASM applied 1/n, but forward+inverse wants no scaling → undo it
        for (let i = 0; i < totalDataLen; i++) resultData[i] = resultData[i]! * n;
      }

      return result;
    }
  }

  // WASM path for non-last axis: gather columns → batch FFT → scatter back.
  // This avoids the JS FFT for the strided axis.
  if (innerSize > 1 && n >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier) {
    const totalRows = outerSize * innerSize;
    const batchDataLen = totalRows * n * 2;

    // Gather: extract strided columns into contiguous batch buffer
    const gathered = new Float64Array(batchDataLen);
    let row = 0;
    for (let outer = 0; outer < outerSize; outer++) {
      for (let inner = 0; inner < innerSize; inner++) {
        const rowOff = row * n * 2;
        for (let k = 0; k < n; k++) {
          const srcIdx = ((outer * n + k) * innerSize + inner) * 2;
          gathered[rowOff + k * 2] = srcData[srcIdx]!;
          gathered[rowOff + k * 2 + 1] = srcData[srcIdx + 1]!;
        }
        row++;
      }
    }

    const outData = wasmFftBatch(gathered, n, totalRows, inverse);
    if (outData) {
      // Scatter: write back from contiguous batch to strided positions
      row = 0;
      for (let outer = 0; outer < outerSize; outer++) {
        for (let inner = 0; inner < innerSize; inner++) {
          const rowOff = row * n * 2;
          for (let k = 0; k < n; k++) {
            const dstIdx = ((outer * n + k) * innerSize + inner) * 2;
            resultData[dstIdx] = outData[rowOff + k * 2]!;
            resultData[dstIdx + 1] = outData[rowOff + k * 2 + 1]!;
          }
          row++;
        }
      }

      // Apply normalization adjustments for non-backward modes
      const totalDataLen = outerSize * n * innerSize * 2;
      if (norm === 'ortho') {
        const scale = inverse ? Math.sqrt(n) : 1 / Math.sqrt(n);
        for (let i = 0; i < totalDataLen; i++) resultData[i] = resultData[i]! * scale;
      } else if (norm === 'forward' && !inverse) {
        const scale = 1 / n;
        for (let i = 0; i < totalDataLen; i++) resultData[i] = resultData[i]! * scale;
      } else if (norm === 'forward' && inverse) {
        for (let i = 0; i < totalDataLen; i++) resultData[i] = resultData[i]! * n;
      }

      return result;
    }
  }

  // JS fallback for small sizes
  const real = new Float64Array(n);
  const imag = new Float64Array(n);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      // Extract slice along axis
      for (let k = 0; k < n; k++) {
        const idx = (outer * n + k) * innerSize + inner;
        real[k] = srcData[idx * 2]!;
        imag[k] = srcData[idx * 2 + 1]!;
      }

      // Perform FFT
      fftCore(real, imag, inverse);

      // Apply normalization
      if (norm === 'ortho') {
        const scale = 1 / Math.sqrt(n);
        if (!inverse) {
          for (let k = 0; k < n; k++) {
            real[k] = real[k]! * scale;
            imag[k] = imag[k]! * scale;
          }
        } else {
          const adjustScale = Math.sqrt(n);
          for (let k = 0; k < n; k++) {
            real[k] = real[k]! * adjustScale;
            imag[k] = imag[k]! * adjustScale;
          }
        }
      } else if (norm === 'forward' && !inverse) {
        const scale = 1 / n;
        for (let k = 0; k < n; k++) {
          real[k] = real[k]! * scale;
          imag[k] = imag[k]! * scale;
        }
      } else if (norm === 'backward' && inverse) {
        // Backward norm (default): fftCore already applies 1/n for inverse, which is correct
      } else if (norm === 'forward' && inverse) {
        for (let k = 0; k < n; k++) {
          real[k] = real[k]! * n;
          imag[k] = imag[k]! * n;
        }
      }

      // Store result
      for (let k = 0; k < n; k++) {
        const idx = (outer * n + k) * innerSize + inner;
        resultData[idx * 2] = real[k]!;
        resultData[idx * 2 + 1] = imag[k]!;
      }
    }
  }

  return result;
}

/**
 * Compute the 1-D FFT of a real-valued array.
 *
 * @param a - Input array storage (real-valued)
 * @param n - Length of the transformed axis
 * @param axis - Axis along which to compute the FFT
 * @param norm - Normalization mode
 * @returns Complex array with length n//2 + 1 along the transformed axis
 */
export function rfft(
  a: ArrayStorage,
  n?: number,
  axis: number = -1,
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  throwIfComplex(a.dtype, 'rfft', 'rfft expects real input.');
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axis
  const ax = axis < 0 ? ndim + axis : axis;

  // Determine input length
  const inputLen = n ?? shape[ax]!;
  const outLen = Math.floor(inputLen / 2) + 1;

  // WASM fast path: 1D rfft on last axis, no truncation/padding, backward norm
  if (
    ndim === 1 &&
    n === undefined &&
    norm === 'backward' &&
    !isComplexDType(a.dtype) &&
    a.dtype === 'float64'
  ) {
    const result = wasmRfft(a, inputLen);
    if (result) return result;
  }

  // WASM fast path: batch rfft when last axis is contiguous and input is real/float
  if (
    ax === ndim - 1 &&
    a.isCContiguous &&
    n === undefined &&
    norm === 'backward' &&
    !isComplexDType(a.dtype) &&
    inputLen >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier &&
    inputLen % 2 === 0
  ) {
    const result = wasmBatchRfft(a, inputLen);
    if (result) return result;
  }

  // Fallback: full FFT + truncation
  const fullResult = fft(a, inputLen, axis, norm);
  const truncated = truncateAxis(fullResult, ax, outLen);
  fullResult.dispose();
  return truncated;
}

/**
 * Compute the inverse of rfft.
 *
 * @param a - Input array storage (from rfft)
 * @param n - Length of the output along the transformed axis
 * @param axis - Axis along which to compute the IFFT
 * @param norm - Normalization mode
 * @returns Real array
 */
export function irfft(
  a: ArrayStorage,
  n?: number,
  axis: number = -1,
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axis
  const ax = axis < 0 ? ndim + axis : axis;

  // Determine output length
  const inputLen = shape[ax]!;
  const outLen = n ?? (inputLen - 1) * 2;

  // WASM fast path: 1D irfft on last axis, backward norm
  if (ndim === 1 && norm === 'backward' && isComplexDType(a.dtype) && a.dtype === 'complex128') {
    const result = wasmIrfft(a, outLen);
    if (result) return result;
  }

  // WASM fast path: batch irfft when last axis is contiguous
  if (
    ax === ndim - 1 &&
    a.isCContiguous &&
    norm === 'backward' &&
    outLen % 2 === 0 &&
    outLen >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier &&
    inputLen === Math.floor(outLen / 2) + 1
  ) {
    const result = wasmBatchIrfft(a, outLen);
    if (result) return result;
  }

  // Reconstruct full spectrum using Hermitian symmetry
  // For real signal: X[k] = conj(X[n-k])
  const fullShape = [...shape];
  fullShape[ax] = outLen;

  const cDtype = fftResultDtype(a.dtype as DType);
  const full = ArrayStorage.zeros(fullShape, cDtype);
  const fullData = full.data as Float64Array | Float32Array;
  const complexA = toComplex(a);
  const srcData = complexA.data as Float64Array | Float32Array;

  const outerSize = shape.slice(0, ax).reduce((acc, s) => acc * s, 1);
  const innerSize = shape.slice(ax + 1).reduce((acc, s) => acc * s, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      // Copy the rfft output directly to positions 0..inputLen-1
      for (let k = 0; k < inputLen; k++) {
        const srcIdx = (outer * inputLen + k) * innerSize + inner;
        const dstIdx = (outer * outLen + k) * innerSize + inner;
        fullData[dstIdx * 2] = srcData[srcIdx * 2]!;
        fullData[dstIdx * 2 + 1] = srcData[srcIdx * 2 + 1]!;
      }

      // Fill in the conjugate symmetric part (indices inputLen to outLen-1)
      // X[outLen - k] = conj(X[k]) for k = 1, 2, ..., outLen - inputLen
      for (let dstK = inputLen; dstK < outLen; dstK++) {
        const srcK = outLen - dstK; // srcK will be outLen - inputLen, outLen - inputLen - 1, ..., 1
        const srcIdx = (outer * inputLen + srcK) * innerSize + inner;
        const dstIdx = (outer * outLen + dstK) * innerSize + inner;
        fullData[dstIdx * 2] = srcData[srcIdx * 2]!;
        fullData[dstIdx * 2 + 1] = -srcData[srcIdx * 2 + 1]!; // Conjugate
      }
    }
  }

  complexA.dispose();

  // Inverse FFT
  const result = ifft(full, outLen, axis, norm);
  full.dispose();

  // Extract real part
  const realResult = extractReal(result);
  result.dispose();
  return realResult;
}

/**
 * Compute the 2-D FFT of a real-valued array.
 */
export function rfft2(
  a: ArrayStorage,
  s?: [number, number],
  axes: [number, number] = [-2, -1],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  throwIfComplex(a.dtype, 'rfft2', 'rfft2 expects real input.');
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  const ax0 = axes[0] < 0 ? ndim + axes[0] : axes[0];
  const ax1 = axes[1] < 0 ? ndim + axes[1] : axes[1];

  // WASM fast path: 2D real input, standard axes, backward norm, no padding
  if (
    ndim === 2 &&
    ax0 === 0 &&
    ax1 === 1 &&
    norm === 'backward' &&
    !s &&
    a.isCContiguous &&
    !isComplexDType(a.dtype)
  ) {
    const rows = shape[0]!;
    const cols = shape[1]!;
    if (
      rows >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier &&
      cols >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier
    ) {
      const halfCols = Math.floor(cols / 2) + 1;
      let inputData: Float64Array;
      if (a.dtype === 'float64') {
        inputData = (a.data as Float64Array).subarray(a.offset, a.offset + rows * cols);
      } else if (a.dtype === 'int64' || a.dtype === 'uint64') {
        const bigData = a.data as BigInt64Array | BigUint64Array;
        inputData = new Float64Array(rows * cols);
        for (let i = 0; i < rows * cols; i++) inputData[i] = Number(bigData[a.offset + i]!);
      } else {
        inputData = Float64Array.from(
          a.data.subarray(a.offset, a.offset + rows * cols) as ArrayLike<number>
        );
      }
      const outData = wasmRfft2(inputData, rows, cols);
      if (outData) {
        const cDtype = fftResultDtype(a.dtype as DType);
        // wasmRfft2 returns Float64Array; convert to Float32Array for complex64
        const typedOut = cDtype === 'complex64' ? Float32Array.from(outData) : outData;
        return ArrayStorage.fromData(typedOut, [rows, halfCols], cDtype);
      }
    }
  }

  // Fallback: use fft2 + truncate
  const result = fft2(a, s, axes, norm);
  const n1 = s ? s[1] : shape[ax1]!;
  const outLen = Math.floor(n1 / 2) + 1;
  return truncateAxis(result, ax1, outLen);
}

/**
 * Compute the inverse of rfft2.
 */
export function irfft2(
  a: ArrayStorage,
  s?: [number, number],
  axes: [number, number] = [-2, -1],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axes
  const ax0 = axes[0] < 0 ? ndim + axes[0] : axes[0];
  const ax1 = axes[1] < 0 ? ndim + axes[1] : axes[1];

  // Determine output lengths
  const inputLen1 = shape[ax1]!;
  const outLen1 = s ? s[1] : (inputLen1 - 1) * 2;
  const outLen0 = s ? s[0] : shape[ax0]!;

  // WASM fast path: 2D complex input, standard axes, backward norm
  if (
    ndim === 2 &&
    ax0 === 0 &&
    ax1 === 1 &&
    norm === 'backward' &&
    !s &&
    a.isCContiguous &&
    isComplexDType(a.dtype)
  ) {
    const rows = shape[0]!;
    const colsHalf = shape[1]!;
    if (
      rows >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier &&
      outLen1 >= FFT_WASM_THRESHOLD * wasmConfig.thresholdMultiplier
    ) {
      const inLen = rows * colsHalf * 2;
      let inputData: Float64Array | Float32Array;
      if (a.dtype === 'complex64') {
        // complex64 WASM-backed data may be Float64Array view — extract to proper Float32Array
        inputData = new Float32Array(inLen);
        const src = a.data;
        const off = a.offset;
        for (let i = 0; i < inLen; i++) inputData[i] = Number(src[off + i]!);
      } else {
        inputData = (a.data as Float64Array).subarray(a.offset, a.offset + inLen);
      }
      const outData = wasmIrfft2(inputData, rows, colsHalf, outLen1);
      if (outData) {
        const outDtype: DType = a.dtype === 'complex64' ? 'float32' : 'float64';
        return ArrayStorage.fromData(outData, [rows, outLen1], outDtype);
      }
    }
  }

  // Fallback: irfft along last axis, then ifft along first axis
  let result = irfft(a, outLen1, ax1, norm);
  result = ifft(result, outLen0, ax0, norm);

  return extractReal(result);
}

/**
 * Compute the N-D FFT of a real-valued array.
 */
export function rfftn(
  a: ArrayStorage,
  s?: number[],
  axes?: number[],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  throwIfComplex(a.dtype, 'rfftn', 'rfftn expects real input.');
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axes
  let axesList: number[];
  if (axes === undefined) {
    axesList = Array.from({ length: ndim }, (_, i) => i);
  } else {
    axesList = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  if (axesList.length === 0) {
    return toComplex(a);
  }

  // Apply rfft along the LAST axis FIRST — while data is still real, this uses
  // the WASM batch rfft kernel (half the FFT work) and produces smaller output.
  const lastAx = axesList[axesList.length - 1]!;
  const lastN = s ? s[axesList.length - 1] : undefined;
  let result = rfft(a, lastN, lastAx, norm);

  // Apply FFT along remaining axes (on the reduced complex data)
  for (let i = 0; i < axesList.length - 1; i++) {
    const ax = axesList[i]!;
    const n = s ? s[i] : undefined;
    const prev = result;
    result = fft(result, n, ax, norm);
    prev.dispose();
  }

  return result;
}

/**
 * Compute the inverse of rfftn.
 */
export function irfftn(
  a: ArrayStorage,
  s?: number[],
  axes?: number[],
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axes
  let axesList: number[];
  if (axes === undefined) {
    axesList = Array.from({ length: ndim }, (_, i) => i);
  } else {
    axesList = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  if (axesList.length === 0) {
    const cplx = toComplex(a);
    const realResult = extractReal(cplx);
    cplx.dispose();
    return realResult;
  }

  // Determine output lengths
  const lastAx = axesList[axesList.length - 1]!;
  const inputLastLen = shape[lastAx]!;
  const outLens = s
    ? [...s]
    : axesList.map((ax, i) => {
        if (i === axesList.length - 1) {
          return (inputLastLen - 1) * 2;
        }
        return shape[ax]!;
      });

  // WASM fast path: fused 3D irfftn in one WASM call
  if (
    ndim === 3 &&
    axesList.length === 3 &&
    axesList[0] === 0 &&
    axesList[1] === 1 &&
    axesList[2] === 2 &&
    norm === 'backward' &&
    s === undefined &&
    a.isCContiguous &&
    isComplexDType(a.dtype) &&
    outLens[2]! % 2 === 0 &&
    (a.dtype === 'complex128' || a.dtype === 'float64')
  ) {
    const d0 = outLens[0]!;
    const d1 = outLens[1]!;
    const d2Half = inputLastLen;
    const d2Out = outLens[2]!;
    const cplx = toComplex(a);
    const outData = wasmIrfftn3d(cplx.data as Float64Array, d0, d1, d2Half, d2Out);
    cplx.dispose();
    if (outData) return ArrayStorage.fromData(outData, [d0, d1, d2Out], 'float64');
  }

  // Apply irfft along the last axis first (recovers full spectrum from Hermitian half)
  let result = irfft(a, outLens[axesList.length - 1], lastAx, norm);

  // Apply ifft along remaining axes (in reverse order for consistency with NumPy)
  for (let i = axesList.length - 2; i >= 0; i--) {
    const ax = axesList[i]!;
    const prev = result;
    result = ifft(result, outLens[i], ax, norm);
    prev.dispose();
  }

  const realResult = extractReal(result);
  result.dispose();
  return realResult;
}

/**
 * Compute the FFT of a signal with Hermitian symmetry (real spectrum).
 *
 * @param a - Input array with Hermitian symmetry
 * @param n - Length of the output
 * @param axis - Axis along which to compute
 * @param norm - Normalization mode
 * @returns Real array
 */
export function hfft(
  a: ArrayStorage,
  n?: number,
  axis: number = -1,
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const ax = axis < 0 ? ndim + axis : axis;

  // Output length
  const inputLen = shape[ax]!;
  const outLen = n ?? (inputLen - 1) * 2;

  // hfft = irfft(conj(a))
  const inputDtype = a.dtype;
  const complexA = toComplex(a);
  const conjA = conjugate(complexA);
  complexA.dispose();
  const result = irfft(conjA, outLen, axis, norm);
  conjA.dispose();

  // Scale by n for correct normalization
  const resultData = result.data as Float64Array | Float32Array;
  for (let i = 0; i < result.size; i++) {
    resultData[i] = resultData[i]! * outLen;
  }

  // NumPy: hfft(float16) → float16
  if (inputDtype === 'float16' && hasFloat16) {
    const f16 = ArrayStorage.empty(Array.from(result.shape), 'float16');
    (f16.data as unknown as Float16Array).set(result.data as Float32Array);
    result.dispose();
    return f16;
  }

  return result;
}

/**
 * Compute the inverse of hfft.
 *
 * @param a - Input real array
 * @param n - Length of the Hermitian output
 * @param axis - Axis along which to compute
 * @param norm - Normalization mode
 * @returns Complex array with Hermitian symmetry
 */
export function ihfft(
  a: ArrayStorage,
  n?: number,
  axis: number = -1,
  norm: 'backward' | 'ortho' | 'forward' = 'backward'
): ArrayStorage {
  throwIfComplex(a.dtype, 'ihfft', 'ihfft expects real input.');
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const ax = axis < 0 ? ndim + axis : axis;

  // Input length
  const inputLen = n ?? shape[ax]!;

  // ihfft = conj(rfft(a))
  const rfftResult = rfft(a, inputLen, axis, norm);

  // Conjugate and scale
  const result = conjugate(rfftResult);
  rfftResult.dispose();
  const resultData = result.data as Float64Array;
  for (let i = 0; i < result.size * 2; i++) {
    resultData[i] = resultData[i]! / inputLen;
  }

  return result;
}

/**
 * Conjugate a complex array
 */
function conjugate(a: ArrayStorage): ArrayStorage {
  const shape = Array.from(a.shape);
  const size = a.size;
  const result = ArrayStorage.zeros(shape, a.dtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data as Float64Array | Float32Array;

  for (let i = 0; i < size; i++) {
    resultData[i * 2] = srcData[i * 2]!;
    resultData[i * 2 + 1] = -srcData[i * 2 + 1]!;
  }

  return result;
}

/**
 * Extract real part from complex array
 */
function extractReal(a: ArrayStorage): ArrayStorage {
  const shape = Array.from(a.shape);
  const size = a.size;
  // complex128 → float64, complex64 → float32
  const realDtype: DType = a.dtype === 'complex64' ? 'float32' : 'float64';
  const result = ArrayStorage.zeros(shape, realDtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data as Float64Array | Float32Array;

  for (let i = 0; i < size; i++) {
    resultData[i] = srcData[i * 2]!;
  }

  return result;
}

/**
 * Return the Discrete Fourier Transform sample frequencies.
 *
 * @param n - Window length
 * @param d - Sample spacing (default: 1.0)
 * @returns Array of length n containing the sample frequencies
 */
export function fftfreq(n: number, d: number = 1.0): ArrayStorage {
  const result = ArrayStorage.zeros([n], 'float64');
  const data = result.data as Float64Array;

  // For even n: positive freqs at indices 0..n/2-1, negative at n/2..n-1
  // For odd n: positive freqs at indices 0..(n-1)/2, negative at (n+1)/2..n-1
  const positiveEnd = Math.floor((n - 1) / 2) + 1; // One past the last positive freq index

  // Positive frequencies: 0, 1, 2, ..., floor((n-1)/2)
  for (let i = 0; i < positiveEnd; i++) {
    data[i] = i / (n * d);
  }

  // Negative frequencies
  for (let i = positiveEnd; i < n; i++) {
    data[i] = (i - n) / (n * d);
  }

  return result;
}

/**
 * Return the Discrete Fourier Transform sample frequencies for rfft.
 *
 * @param n - Window length
 * @param d - Sample spacing (default: 1.0)
 * @returns Array of length n//2 + 1 containing the sample frequencies
 */
export function rfftfreq(n: number, d: number = 1.0): ArrayStorage {
  const outLen = Math.floor(n / 2) + 1;
  const result = ArrayStorage.zeros([outLen], 'float64');
  const data = result.data as Float64Array;

  for (let i = 0; i < outLen; i++) {
    data[i] = i / (n * d);
  }

  return result;
}

/**
 * Shift the zero-frequency component to the center of the spectrum.
 *
 * @param a - Input array
 * @param axes - Axes over which to shift (default: all axes)
 * @returns Shifted array
 */
export function fftshift(a: ArrayStorage, axes?: number | number[]): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axes
  let axesList: number[];
  if (axes === undefined) {
    axesList = Array.from({ length: ndim }, (_, i) => i);
  } else if (typeof axes === 'number') {
    axesList = [axes < 0 ? ndim + axes : axes];
  } else {
    axesList = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  // Calculate shift amounts — only for the specified axes
  const shifts = axesList.map((ax) => Math.floor(shape[ax]! / 2));

  // For 1D arrays shifting all axes, use flat roll (hits WASM fast path)
  if (ndim === 1 && axesList.length === 1 && axesList[0] === 0) {
    return shapeRoll(a, shifts[0]!);
  }
  return shapeRoll(a, shifts, axesList);
}

/**
 * Inverse of fftshift.
 *
 * @param a - Input array
 * @param axes - Axes over which to shift (default: all axes)
 * @returns Shifted array
 */
export function ifftshift(a: ArrayStorage, axes?: number | number[]): ArrayStorage {
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axes
  let axesList: number[];
  if (axes === undefined) {
    axesList = Array.from({ length: ndim }, (_, i) => i);
  } else if (typeof axes === 'number') {
    axesList = [axes < 0 ? ndim + axes : axes];
  } else {
    axesList = axes.map((ax) => (ax < 0 ? ndim + ax : ax));
  }

  // Calculate shift amounts (negative of fftshift) — only for the specified axes
  const shifts = axesList.map((ax) => -Math.floor(shape[ax]! / 2));

  // For 1D arrays shifting all axes, use flat roll (hits WASM fast path)
  if (ndim === 1 && axesList.length === 1 && axesList[0] === 0) {
    return shapeRoll(a, shifts[0]!);
  }
  return shapeRoll(a, shifts, axesList);
}
