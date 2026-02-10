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
import type { DType } from '../dtype';

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

  // Handle empty arrays
  if (a.size === 0) {
    return ArrayStorage.zeros(shape, 'complex128');
  }

  // Handle 0-D arrays
  if (ndim === 0) {
    const result = ArrayStorage.zeros([1], 'complex128');
    const val = a.iget(0);
    const re = val instanceof Complex ? val.re : Number(val);
    const im = val instanceof Complex ? val.im : 0;
    const data = result.data as Float64Array;
    data[0] = re;
    data[1] = im;
    // Reshape to scalar
    return ArrayStorage.fromData(result.data, [], 'complex128');
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
    result = padOrTruncate(result, outShape, axesList);
  }

  // Apply FFT along each axis
  for (const axis of axesList) {
    result = fft1dAlongAxis(result, axis, inverse, norm);
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

  if (dtype === 'complex128' || dtype === 'complex64') {
    // Already complex, just copy
    const result = ArrayStorage.zeros(shape, 'complex128');
    const resultData = result.data as Float64Array;
    const srcData = a.data as Float64Array | Float32Array;

    // Use optimized copy for complex data (interleaved real/imag)
    for (let i = 0; i < size * 2; i++) {
      resultData[i] = srcData[i]!;
    }
    return result;
  }

  // Real array: convert to complex using direct typed array access
  const result = ArrayStorage.zeros(shape, 'complex128');
  const resultData = result.data as Float64Array;
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

  const result = ArrayStorage.zeros(shape, 'complex128');
  const resultData = result.data as Float64Array;
  const srcData = a.data as Float64Array;

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

  const result = ArrayStorage.zeros(shape, 'complex128');
  const resultData = result.data as Float64Array;
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
function fft1dAlongAxis(
  a: ArrayStorage,
  axis: number,
  inverse: boolean,
  norm: 'backward' | 'ortho' | 'forward'
): ArrayStorage {
  const shape = Array.from(a.shape);
  const n = shape[axis]!;

  if (n === 0) return a;

  const result = ArrayStorage.zeros(shape, 'complex128');
  const resultData = result.data as Float64Array;
  const srcData = a.data as Float64Array;

  // Calculate dimensions
  const outerSize = shape.slice(0, axis).reduce((acc, s) => acc * s, 1);
  const innerSize = shape.slice(axis + 1).reduce((acc, s) => acc * s, 1);

  // Temporary arrays for FFT
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
          // For forward FFT with ortho, we don't apply 1/n in fftCore, so apply 1/sqrt(n)
          for (let k = 0; k < n; k++) {
            real[k] = real[k]! * scale;
            imag[k] = imag[k]! * scale;
          }
        } else {
          // For inverse FFT with ortho, fftCore already applied 1/n, so multiply by sqrt(n)
          const adjustScale = Math.sqrt(n);
          for (let k = 0; k < n; k++) {
            real[k] = real[k]! * adjustScale;
            imag[k] = imag[k]! * adjustScale;
          }
        }
      } else if (norm === 'forward' && !inverse) {
        // Forward norm: apply 1/n for forward FFT
        const scale = 1 / n;
        for (let k = 0; k < n; k++) {
          real[k] = real[k]! * scale;
          imag[k] = imag[k]! * scale;
        }
      } else if (norm === 'backward' && inverse) {
        // Backward norm (default): fftCore already applies 1/n for inverse, which is correct
      } else if (norm === 'forward' && inverse) {
        // Forward norm with inverse: no scaling (undo the 1/n from fftCore)
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
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axis
  const ax = axis < 0 ? ndim + axis : axis;

  // Determine input length
  const inputLen = n ?? shape[ax]!;

  // Full FFT
  const fullResult = fft(a, inputLen, axis, norm);

  // Take only first n//2 + 1 elements along axis
  const outLen = Math.floor(inputLen / 2) + 1;

  return truncateAxis(fullResult, ax, outLen);
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

  // Reconstruct full spectrum using Hermitian symmetry
  // For real signal: X[k] = conj(X[n-k])
  const fullShape = [...shape];
  fullShape[ax] = outLen;

  const full = ArrayStorage.zeros(fullShape, 'complex128');
  const fullData = full.data as Float64Array;
  const srcData = toComplex(a).data as Float64Array;

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

  // Inverse FFT
  const result = ifft(full, outLen, axis, norm);

  // Extract real part
  return extractReal(result);
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
  const shape = Array.from(a.shape);
  const ndim = shape.length;

  // Normalize axes
  const ax0 = axes[0] < 0 ? ndim + axes[0] : axes[0];
  const ax1 = axes[1] < 0 ? ndim + axes[1] : axes[1];

  // Apply FFT along first axis
  let result = fft(a, s ? s[0] : undefined, ax0, norm);

  // Apply rFFT along second axis
  const outShape = Array.from(result.shape);
  const n1 = s ? s[1] : outShape[ax1]!;
  result = fft(result, n1, ax1, norm);

  // Truncate last axis to n1//2 + 1
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

  // Apply irfft along last axis first
  let result = irfft(a, outLen1, ax1, norm);

  // Apply ifft along first axis
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

  // Apply FFT along all axes except the last
  let result: ArrayStorage = a;
  for (let i = 0; i < axesList.length - 1; i++) {
    const ax = axesList[i]!;
    const n = s ? s[i] : undefined;
    result = fft(result, n, ax, norm);
  }

  // Apply rfft along the last axis
  const lastAx = axesList[axesList.length - 1]!;
  const lastN = s ? s[axesList.length - 1] : undefined;
  return rfft(result, lastN, lastAx, norm);
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
    return extractReal(toComplex(a));
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

  // Apply irfft along the last axis first
  let result = irfft(a, outLens[axesList.length - 1], lastAx, norm);

  // Apply ifft along remaining axes (in reverse order for consistency with NumPy)
  for (let i = axesList.length - 2; i >= 0; i--) {
    const ax = axesList[i]!;
    result = ifft(result, outLens[i], ax, norm);
  }

  return extractReal(result);
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
  const conjA = conjugate(toComplex(a));
  const result = irfft(conjA, outLen, axis, norm);

  // Scale by n for correct normalization
  const resultData = result.data as Float64Array;
  for (let i = 0; i < result.size; i++) {
    resultData[i] = resultData[i]! * outLen;
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
  const shape = Array.from(a.shape);
  const ndim = shape.length;
  const ax = axis < 0 ? ndim + axis : axis;

  // Input length
  const inputLen = n ?? shape[ax]!;

  // ihfft = conj(rfft(a))
  const rfftResult = rfft(a, inputLen, axis, norm);

  // Conjugate and scale
  const result = conjugate(rfftResult);
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
  const result = ArrayStorage.zeros(shape, 'complex128');
  const resultData = result.data as Float64Array;
  const srcData = a.data as Float64Array;

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
  const result = ArrayStorage.zeros(shape, 'float64');
  const resultData = result.data as Float64Array;
  const srcData = a.data as Float64Array;

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

  // Calculate shift amounts
  const shifts = shape.map((_, i) => (axesList.includes(i) ? Math.floor(shape[i]! / 2) : 0));

  return roll(a, shifts);
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

  // Calculate shift amounts (negative of fftshift)
  const shifts = shape.map((_, i) => (axesList.includes(i) ? -Math.floor(shape[i]! / 2) : 0));

  return roll(a, shifts);
}

/**
 * Roll array elements along specified axes.
 */
function roll(a: ArrayStorage, shifts: number[]): ArrayStorage {
  const shape = Array.from(a.shape);
  const dtype = a.dtype as DType;
  const size = a.size;

  // Create result array with same dtype
  const result = ArrayStorage.zeros(shape, dtype);
  const isComplex = dtype === 'complex128' || dtype === 'complex64';

  // Calculate strides for index conversion
  const strides = new Array(shape.length);
  strides[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1]! * shape[i + 1]!;
  }

  // Copy with shifts
  for (let flatIdx = 0; flatIdx < size; flatIdx++) {
    // Convert flat index to multi-dimensional index
    const srcIndices = new Array(shape.length);
    let remainder = flatIdx;
    for (let d = 0; d < shape.length; d++) {
      srcIndices[d] = Math.floor(remainder / strides[d]!);
      remainder = remainder % strides[d]!;
    }

    // Apply shifts with wrapping
    const dstIndices = srcIndices.map((idx, d) => {
      let newIdx = idx + shifts[d]!;
      const dim = shape[d]!;
      newIdx = ((newIdx % dim) + dim) % dim;
      return newIdx;
    });

    // Convert back to flat index
    let dstFlatIdx = 0;
    for (let d = 0; d < shape.length; d++) {
      dstFlatIdx += dstIndices[d]! * strides[d]!;
    }

    // Copy value(s)
    if (isComplex) {
      const srcData = a.data as Float64Array;
      const dstData = result.data as Float64Array;
      dstData[dstFlatIdx * 2] = srcData[flatIdx * 2]!;
      dstData[dstFlatIdx * 2 + 1] = srcData[flatIdx * 2 + 1]!;
    } else {
      const val = a.iget(flatIdx);
      result.iset(dstFlatIdx, val);
    }
  }

  return result;
}
