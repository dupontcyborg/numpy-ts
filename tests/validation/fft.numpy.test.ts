/**
 * Python NumPy validation tests for FFT operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, fft, Complex } from '../../src/index';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: FFT Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          '   Current Python command: ' +
          (process.env.NUMPY_PYTHON || 'python3') +
          '\n'
      );
    }
  });

  describe('fft.fft', () => {
    it('matches NumPy for 1D real array', () => {
      const jsResult = fft.fft(array([1, 2, 3, 4, 5, 6, 7, 8]));
      const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for prime-length array', () => {
      const jsResult = fft.fft(array([1, 2, 3, 4, 5, 6, 7]));
      const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with n parameter', () => {
      const jsResult = fft.fft(array([1, 2, 3, 4]), 8);
      const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4]), n=8)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for complex input', () => {
      const jsResult = fft.fft(
        array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3), new Complex(4, 4)])
      );
      const pyResult = runNumPy(`
result = np.fft.fft(np.array([1+1j, 2+2j, 3+3j, 4+4j]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.ifft', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = fft.ifft(array([36, -4, -4, -4, -4, -4, -4, -4]));
      const pyResult = runNumPy(`
result = np.fft.ifft(np.array([36, -4, -4, -4, -4, -4, -4, -4]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('is inverse of fft (matches NumPy)', () => {
      const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const transformed = fft.fft(original);
      const recovered = fft.ifft(transformed);
      const pyResult = runNumPy(`
original = np.array([1, 2, 3, 4, 5, 6, 7, 8])
result = np.fft.ifft(np.fft.fft(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.fft2', () => {
    it('matches NumPy for 2D array', () => {
      const jsResult = fft.fft2(
        array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.fft.fft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.ifft2', () => {
    it('is inverse of fft2 (matches NumPy)', () => {
      const original = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const recovered = fft.ifft2(fft.fft2(original));
      const pyResult = runNumPy(`
original = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = np.fft.ifft2(np.fft.fft2(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.fftn', () => {
    it('matches NumPy for 3D array', () => {
      const jsResult = fft.fftn(
        array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ])
      );
      const pyResult = runNumPy(`
result = np.fft.fftn(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.ifftn', () => {
    it('is inverse of fftn (matches NumPy)', () => {
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
      const recovered = fft.ifftn(fft.fftn(original));
      const pyResult = runNumPy(`
original = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.fft.ifftn(np.fft.fftn(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.rfft', () => {
    it('matches NumPy for 1D real array', () => {
      const jsResult = fft.rfft(array([1, 2, 3, 4, 5, 6, 7, 8]));
      const pyResult = runNumPy(`
result = np.fft.rfft(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with n parameter', () => {
      const jsResult = fft.rfft(array([1, 2, 3, 4]), 8);
      const pyResult = runNumPy(`
result = np.fft.rfft(np.array([1, 2, 3, 4]), n=8)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.irfft', () => {
    it('matches NumPy roundtrip', () => {
      const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
      const spectrum = fft.rfft(original);
      const recovered = fft.irfft(spectrum);
      const pyResult = runNumPy(`
original = np.array([1, 2, 3, 4, 5, 6, 7, 8])
result = np.fft.irfft(np.fft.rfft(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.rfft2', () => {
    it('matches NumPy for 2D real array', () => {
      const jsResult = fft.rfft2(
        array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.fft.rfft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.irfft2', () => {
    it('matches NumPy roundtrip', () => {
      const original = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const recovered = fft.irfft2(fft.rfft2(original));
      const pyResult = runNumPy(`
original = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = np.fft.irfft2(np.fft.rfft2(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.rfftn', () => {
    it('matches NumPy for 3D real array', () => {
      const jsResult = fft.rfftn(
        array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ])
      );
      const pyResult = runNumPy(`
result = np.fft.rfftn(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.irfftn', () => {
    it('matches NumPy roundtrip', () => {
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
      const recovered = fft.irfftn(fft.rfftn(original));
      const pyResult = runNumPy(`
original = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = np.fft.irfftn(np.fft.rfftn(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.fftfreq', () => {
    it('matches NumPy for even n', () => {
      const jsResult = fft.fftfreq(8);
      const pyResult = runNumPy(`
result = np.fft.fftfreq(8)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for odd n', () => {
      const jsResult = fft.fftfreq(7);
      const pyResult = runNumPy(`
result = np.fft.fftfreq(7)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with custom sample spacing', () => {
      const jsResult = fft.fftfreq(8, 0.1);
      const pyResult = runNumPy(`
result = np.fft.fftfreq(8, d=0.1)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.rfftfreq', () => {
    it('matches NumPy for even n', () => {
      const jsResult = fft.rfftfreq(8);
      const pyResult = runNumPy(`
result = np.fft.rfftfreq(8)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for odd n', () => {
      const jsResult = fft.rfftfreq(7);
      const pyResult = runNumPy(`
result = np.fft.rfftfreq(7)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.fftshift', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = fft.fftshift(array([0, 1, 2, 3, 4, -4, -3, -2, -1]));
      const pyResult = runNumPy(`
result = np.fft.fftshift(np.array([0, 1, 2, 3, 4, -4, -3, -2, -1]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = fft.fftshift(
        array([
          [0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 10, 11],
          [12, 13, 14, 15],
        ])
      );
      const pyResult = runNumPy(`
result = np.fft.fftshift(np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fft.ifftshift', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = fft.ifftshift(array([-4, -3, -2, -1, 0, 1, 2, 3, 4]));
      const pyResult = runNumPy(`
result = np.fft.ifftshift(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('is inverse of fftshift (matches NumPy)', () => {
      const original = array([0, 1, 2, 3, 4, 5, 6, 7]);
      const shifted = fft.fftshift(original);
      const recovered = fft.ifftshift(shifted);
      const pyResult = runNumPy(`
original = np.array([0, 1, 2, 3, 4, 5, 6, 7])
result = np.fft.ifftshift(np.fft.fftshift(original))
      `);

      expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('normalization modes', () => {
    it('matches NumPy with ortho normalization', () => {
      const jsResult = fft.fft(array([1, 2, 3, 4]), undefined, undefined, 'ortho');
      const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4]), norm='ortho')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with forward normalization', () => {
      const jsResult = fft.fft(array([1, 2, 3, 4]), undefined, undefined, 'forward');
      const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4]), norm='forward')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });
});
