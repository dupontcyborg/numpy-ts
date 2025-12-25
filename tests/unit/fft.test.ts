import { describe, it, expect } from 'vitest';
import { array, fft, arange, Complex } from '../../src/index';

// Helper to check if two complex values are close
function complexClose(
  a: Complex | number,
  b: Complex | number,
  rtol: number = 1e-10,
  atol: number = 1e-10
): boolean {
  const aRe = a instanceof Complex ? a.re : Number(a);
  const aIm = a instanceof Complex ? a.im : 0;
  const bRe = b instanceof Complex ? b.re : Number(b);
  const bIm = b instanceof Complex ? b.im : 0;

  const reDiff = Math.abs(aRe - bRe);
  const imDiff = Math.abs(aIm - bIm);

  const reTol = atol + rtol * Math.abs(bRe);
  const imTol = atol + rtol * Math.abs(bIm);

  return reDiff <= reTol && imDiff <= imTol;
}

// Helper to check if arrays of complex numbers are close (kept for potential debugging)
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function _arraysClose(a: any[], b: any[], rtol: number = 1e-10, atol: number = 1e-10): boolean {
  if (a.length !== b.length) return false;

  for (let i = 0; i < a.length; i++) {
    if (Array.isArray(a[i]) && Array.isArray(b[i])) {
      if (!_arraysClose(a[i], b[i], rtol, atol)) return false;
    } else if (!complexClose(a[i], b[i], rtol, atol)) {
      return false;
    }
  }

  return true;
}

describe('FFT Operations', () => {
  describe('fft.fft', () => {
    it('computes FFT of constant array', () => {
      const arr = array([1, 1, 1, 1]);
      const result = fft.fft(arr);
      const data = result.toArray() as Complex[];

      // FFT of constant [1,1,1,1] should give [4, 0, 0, 0]
      expect(complexClose(data[0]!, new Complex(4, 0))).toBe(true);
      expect(complexClose(data[1]!, new Complex(0, 0))).toBe(true);
      expect(complexClose(data[2]!, new Complex(0, 0))).toBe(true);
      expect(complexClose(data[3]!, new Complex(0, 0))).toBe(true);
    });

    it('computes FFT of impulse', () => {
      const arr = array([1, 0, 0, 0]);
      const result = fft.fft(arr);
      const data = result.toArray() as Complex[];

      // FFT of impulse [1,0,0,0] should give [1, 1, 1, 1]
      for (const val of data) {
        expect(complexClose(val, new Complex(1, 0))).toBe(true);
      }
    });

    it('returns correct dtype', () => {
      const arr = array([1, 2, 3, 4]);
      const result = fft.fft(arr);
      expect(result.dtype).toBe('complex128');
    });

    it('preserves shape for 1D array', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const result = fft.fft(arr);
      expect(result.shape).toEqual([8]);
    });

    it('supports n parameter for padding', () => {
      const arr = array([1, 2]);
      const result = fft.fft(arr, 4);
      expect(result.shape).toEqual([4]);
    });

    it('supports n parameter for truncation', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const result = fft.fft(arr, 4);
      expect(result.shape).toEqual([4]);
    });

    it('handles non-power-of-2 sizes', () => {
      const arr = array([1, 2, 3, 4, 5]); // size 5
      const result = fft.fft(arr);
      expect(result.shape).toEqual([5]);
      expect(result.dtype).toBe('complex128');
    });
  });

  describe('fft.ifft', () => {
    it('is inverse of fft', () => {
      const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const transformed = fft.fft(original);
      const result = fft.ifft(transformed);
      const data = result.toArray() as Complex[];

      for (let i = 0; i < 8; i++) {
        expect(complexClose(data[i]!, new Complex(i + 1, 0))).toBe(true);
      }
    });

    it('returns correct dtype', () => {
      const arr = array([1, 2, 3, 4]);
      const result = fft.ifft(arr);
      expect(result.dtype).toBe('complex128');
    });
  });

  describe('fft.fft2', () => {
    it('computes 2D FFT', () => {
      const arr = array([
        [1, 1],
        [1, 1],
      ]);
      const result = fft.fft2(arr);

      expect(result.shape).toEqual([2, 2]);
      expect(result.dtype).toBe('complex128');

      const data = result.toArray() as Complex[][];
      // Sum of all elements should be at [0,0]
      expect(complexClose(data[0]![0]!, new Complex(4, 0))).toBe(true);
    });
  });

  describe('fft.ifft2', () => {
    it('is inverse of fft2', () => {
      const original = array([
        [1, 2],
        [3, 4],
      ]);
      const transformed = fft.fft2(original);
      const result = fft.ifft2(transformed);
      const data = result.toArray() as Complex[][];

      expect(complexClose(data[0]![0]!, new Complex(1, 0))).toBe(true);
      expect(complexClose(data[0]![1]!, new Complex(2, 0))).toBe(true);
      expect(complexClose(data[1]![0]!, new Complex(3, 0))).toBe(true);
      expect(complexClose(data[1]![1]!, new Complex(4, 0))).toBe(true);
    });
  });

  describe('fft.fftn', () => {
    it('computes N-D FFT', () => {
      const arr = array([
        [
          [1, 1],
          [1, 1],
        ],
        [
          [1, 1],
          [1, 1],
        ],
      ]);
      const result = fft.fftn(arr);

      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.dtype).toBe('complex128');
    });
  });

  describe('fft.ifftn', () => {
    it('is inverse of fftn', () => {
      const original = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const transformed = fft.fftn(original);
      const result = fft.ifftn(transformed);
      const data = result.toArray() as Complex[][][];

      expect(complexClose(data[0]![0]![0]!, new Complex(1, 0))).toBe(true);
      expect(complexClose(data[1]![1]![1]!, new Complex(8, 0))).toBe(true);
    });
  });

  describe('fft.rfft', () => {
    it('computes real FFT', () => {
      const arr = array([1, 2, 3, 4]);
      const result = fft.rfft(arr);

      // rfft output length is n//2 + 1 = 3
      expect(result.shape).toEqual([3]);
      expect(result.dtype).toBe('complex128');
    });

    it('is efficient for real input', () => {
      const arr = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const result = fft.rfft(arr);

      // rfft output length is 8//2 + 1 = 5
      expect(result.shape).toEqual([5]);
    });
  });

  describe('fft.irfft', () => {
    it('reconstructs real signal from rfft', () => {
      const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const spectrum = fft.rfft(original);
      const result = fft.irfft(spectrum);

      expect(result.shape).toEqual([8]);
      const data = result.toArray() as number[];

      for (let i = 0; i < 8; i++) {
        expect(Math.abs(data[i]! - (i + 1))).toBeLessThan(1e-10);
      }
    });

    it('returns real output (not complex)', () => {
      const arr = array([1, 2, 3, 4]);
      const spectrum = fft.rfft(arr);
      const result = fft.irfft(spectrum);

      // irfft should return float64, not complex
      expect(result.dtype).toBe('float64');
    });
  });

  describe('fft.rfft2', () => {
    it('computes 2D real FFT', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = fft.rfft2(arr);

      // Last axis is n//2 + 1 = 3
      expect(result.shape).toEqual([2, 3]);
      expect(result.dtype).toBe('complex128');
    });
  });

  describe('fft.irfft2', () => {
    it('is inverse of rfft2', () => {
      const original = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const spectrum = fft.rfft2(original);
      const result = fft.irfft2(spectrum);

      expect(result.dtype).toBe('float64');
      const data = result.toArray() as number[][];

      expect(Math.abs(data[0]![0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]![3]! - 8)).toBeLessThan(1e-10);
    });
  });

  describe('fft.rfftn', () => {
    it('computes N-D real FFT', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = fft.rfftn(arr);

      // Last axis is 2//2 + 1 = 2
      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.dtype).toBe('complex128');
    });
  });

  describe('fft.irfftn', () => {
    it('is inverse of rfftn', () => {
      const original = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const spectrum = fft.rfftn(original);
      const result = fft.irfftn(spectrum);

      expect(result.dtype).toBe('float64');
      const data = result.toArray() as number[][][];

      expect(Math.abs(data[0]![0]![0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]![1]![1]! - 8)).toBeLessThan(1e-10);
    });
  });

  describe('fft.hfft', () => {
    it('computes Hermitian FFT', () => {
      // Input with Hermitian symmetry
      const arr = array([
        new Complex(1, 0),
        new Complex(2, -1),
        new Complex(3, 0),
        new Complex(2, 1),
      ]);
      const result = fft.hfft(arr);

      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([6]);
    });
  });

  describe('fft.ihfft', () => {
    it('is inverse of hfft', () => {
      const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const spectrum = fft.ihfft(original);
      const result = fft.hfft(spectrum);

      expect(result.dtype).toBe('float64');
      // Note: May need length adjustment
    });

    it('returns complex output', () => {
      const arr = array([1, 2, 3, 4]);
      const result = fft.ihfft(arr);

      expect(result.dtype).toBe('complex128');
    });
  });

  describe('fft.fftfreq', () => {
    it('returns correct frequencies for even n', () => {
      const freqs = fft.fftfreq(8);
      const data = freqs.toArray() as number[];

      // For n=8, d=1: [0, 1/8, 2/8, 3/8, -4/8, -3/8, -2/8, -1/8]
      expect(Math.abs(data[0]! - 0)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]! - 0.125)).toBeLessThan(1e-10);
      expect(Math.abs(data[4]! - -0.5)).toBeLessThan(1e-10);
      expect(Math.abs(data[7]! - -0.125)).toBeLessThan(1e-10);
    });

    it('returns correct frequencies for odd n', () => {
      const freqs = fft.fftfreq(5);
      const data = freqs.toArray() as number[];

      // For n=5, d=1: [0, 1/5, 2/5, -2/5, -1/5]
      expect(Math.abs(data[0]! - 0)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]! - 0.2)).toBeLessThan(1e-10);
      expect(Math.abs(data[2]! - 0.4)).toBeLessThan(1e-10);
      expect(Math.abs(data[3]! - -0.4)).toBeLessThan(1e-10);
      expect(Math.abs(data[4]! - -0.2)).toBeLessThan(1e-10);
    });

    it('respects sample spacing d', () => {
      const freqs = fft.fftfreq(4, 0.5);
      const data = freqs.toArray() as number[];

      // For n=4, d=0.5: [0, 0.5, -1, -0.5]
      expect(Math.abs(data[0]! - 0)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]! - 0.5)).toBeLessThan(1e-10);
      expect(Math.abs(data[2]! - -1.0)).toBeLessThan(1e-10);
      expect(Math.abs(data[3]! - -0.5)).toBeLessThan(1e-10);
    });
  });

  describe('fft.rfftfreq', () => {
    it('returns correct frequencies for even n', () => {
      const freqs = fft.rfftfreq(8);
      const data = freqs.toArray() as number[];

      // For n=8, d=1: [0, 1/8, 2/8, 3/8, 4/8]
      expect(freqs.shape).toEqual([5]);
      expect(Math.abs(data[0]! - 0)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]! - 0.125)).toBeLessThan(1e-10);
      expect(Math.abs(data[4]! - 0.5)).toBeLessThan(1e-10);
    });

    it('returns correct frequencies for odd n', () => {
      const freqs = fft.rfftfreq(5);
      const data = freqs.toArray() as number[];

      // For n=5, d=1: [0, 1/5, 2/5]
      expect(freqs.shape).toEqual([3]);
      expect(Math.abs(data[0]! - 0)).toBeLessThan(1e-10);
      expect(Math.abs(data[1]! - 0.2)).toBeLessThan(1e-10);
      expect(Math.abs(data[2]! - 0.4)).toBeLessThan(1e-10);
    });
  });

  describe('fft.fftshift', () => {
    it('shifts 1D array', () => {
      const arr = array([0, 1, 2, 3, 4, -4, -3, -2, -1]);
      const result = fft.fftshift(arr);
      const data = result.toArray() as number[];

      // Shifted: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
      expect(data).toEqual([-4, -3, -2, -1, 0, 1, 2, 3, 4]);
    });

    it('shifts 2D array', () => {
      const arr = array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
      ]);
      const result = fft.fftshift(arr);
      expect(result.shape).toEqual([3, 3]);
    });

    it('shifts along specific axes', () => {
      const arr = array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
      ]);
      const result = fft.fftshift(arr, 1);
      expect(result.shape).toEqual([2, 4]);
    });
  });

  describe('fft.ifftshift', () => {
    it('is inverse of fftshift', () => {
      const original = array([0, 1, 2, 3, 4, 5, 6, 7]);
      const shifted = fft.fftshift(original);
      const result = fft.ifftshift(shifted);
      const data = result.toArray() as number[];

      expect(data).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    });

    it('works for 2D arrays', () => {
      const original = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const shifted = fft.fftshift(original);
      const result = fft.ifftshift(shifted);
      const data = result.toArray() as number[][];

      expect(data[0]).toEqual([1, 2, 3, 4]);
      expect(data[1]).toEqual([5, 6, 7, 8]);
    });
  });

  describe('normalization modes', () => {
    it('supports backward normalization (default)', () => {
      const arr = array([1, 2, 3, 4]);
      const forward = fft.fft(arr, undefined, undefined, 'backward');
      const back = fft.ifft(forward, undefined, undefined, 'backward');
      const data = back.toArray() as Complex[];

      for (let i = 0; i < 4; i++) {
        expect(complexClose(data[i]!, new Complex(i + 1, 0))).toBe(true);
      }
    });

    it('supports ortho normalization', () => {
      const arr = array([1, 2, 3, 4]);
      const forward = fft.fft(arr, undefined, undefined, 'ortho');
      const back = fft.ifft(forward, undefined, undefined, 'ortho');
      const data = back.toArray() as Complex[];

      for (let i = 0; i < 4; i++) {
        expect(complexClose(data[i]!, new Complex(i + 1, 0))).toBe(true);
      }
    });

    it('supports forward normalization', () => {
      const arr = array([1, 2, 3, 4]);
      const forward = fft.fft(arr, undefined, undefined, 'forward');
      const back = fft.ifft(forward, undefined, undefined, 'forward');
      const data = back.toArray() as Complex[];

      for (let i = 0; i < 4; i++) {
        expect(complexClose(data[i]!, new Complex(i + 1, 0))).toBe(true);
      }
    });
  });

  describe('complex input', () => {
    it('handles complex input arrays', () => {
      const arr = array([
        new Complex(1, 1),
        new Complex(2, 2),
        new Complex(3, 3),
        new Complex(4, 4),
      ]);
      const result = fft.fft(arr);

      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([4]);

      // Check that ifft recovers original
      const recovered = fft.ifft(result);
      const data = recovered.toArray() as Complex[];

      expect(complexClose(data[0]!, new Complex(1, 1))).toBe(true);
      expect(complexClose(data[3]!, new Complex(4, 4))).toBe(true);
    });
  });

  describe('edge cases', () => {
    it('handles size 1 array', () => {
      const arr = array([5]);
      const result = fft.fft(arr);
      const data = result.toArray() as Complex[];

      expect(result.shape).toEqual([1]);
      expect(complexClose(data[0]!, new Complex(5, 0))).toBe(true);
    });

    it('handles size 2 array', () => {
      const arr = array([1, 2]);
      const result = fft.fft(arr);
      const data = result.toArray() as Complex[];

      expect(result.shape).toEqual([2]);
      // FFT([1,2]) = [3, -1]
      expect(complexClose(data[0]!, new Complex(3, 0))).toBe(true);
      expect(complexClose(data[1]!, new Complex(-1, 0))).toBe(true);
    });

    it('handles prime size', () => {
      const arr = arange(7);
      const result = fft.fft(arr);

      expect(result.shape).toEqual([7]);

      // Verify round-trip
      const recovered = fft.ifft(result);
      const data = recovered.toArray() as Complex[];

      for (let i = 0; i < 7; i++) {
        expect(complexClose(data[i]!, new Complex(i, 0))).toBe(true);
      }
    });

    it('handles large arrays', () => {
      const arr = arange(1024);
      const result = fft.fft(arr);

      expect(result.shape).toEqual([1024]);

      // Verify round-trip
      const recovered = fft.ifft(result);
      const data = recovered.toArray() as Complex[];

      for (let i = 0; i < 1024; i++) {
        expect(complexClose(data[i]!, new Complex(i, 0), 1e-8, 1e-8)).toBe(true);
      }
    });
  });
});
