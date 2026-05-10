/**
 * Python NumPy validation tests for FFT operations.
 *
 * Tests all FFT functions across multiple dtypes and WASM execution modes
 * (forced WASM, forced JS fallback) to ensure correctness of both code paths.
 */

import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { array, Complex, fft, wasmConfig } from '../../src/index';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

const WASM_MODES = [
  { name: 'forced WASM (threshold=0)', multiplier: 0 },
  { name: 'forced JS (threshold=Infinity)', multiplier: Infinity },
] as const;

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: FFT Operations [${mode.name}]`, () => {
    beforeAll(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      if (!checkNumPyAvailable()) {
        throw new Error('❌ Python NumPy not available!');
      }
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
    });

    // ========================================
    // fft / ifft
    // ========================================

    describe('fft.fft', () => {
      it('matches NumPy for float64 pow2', () => {
        const jsResult = fft.fft(array([1, 2, 3, 4, 5, 6, 7, 8]));
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float32', () => {
        const a = array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float32');
        const jsResult = fft.fft(a);
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float16', () => {
        const a = array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float16');
        const jsResult = fft.fft(a);
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float16))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for complex128', () => {
        const jsResult = fft.fft(
          array([new Complex(1, 1), new Complex(2, -1), new Complex(3, 0), new Complex(4, 2)]),
        );
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1+1j, 2-1j, 3+0j, 4+2j]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for prime length (Bluestein)', () => {
        const jsResult = fft.fft(array([1, 2, 3, 4, 5, 6, 7]));
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7], dtype=float))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for mixed-radix N=1000', () => {
        const data = Array.from({ length: 1000 }, (_, i) => i % 17);
        const jsResult = fft.fft(array(data));
        const pyResult = runNumPy(`
result = np.fft.fft((np.arange(1000) % 17).astype(float))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy with n parameter (zero-pad)', () => {
        const jsResult = fft.fft(array([1, 2, 3, 4]), 8);
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4]), n=8)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.ifft', () => {
      it('roundtrip matches NumPy for float64', () => {
        const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
        const recovered = fft.ifft(fft.fft(original));
        const pyResult = runNumPy(`
result = np.fft.ifft(np.fft.fft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });

      it('roundtrip matches NumPy for mixed-radix N=100', () => {
        const data = Array.from({ length: 100 }, (_, i) => i * 0.1);
        const original = array(data);
        const recovered = fft.ifft(fft.fft(original));
        const pyResult = runNumPy(`
result = np.fft.ifft(np.fft.fft(np.arange(100) * 0.1))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // rfft / irfft
    // ========================================

    describe('fft.rfft', () => {
      it('matches NumPy for float64', () => {
        const jsResult = fft.rfft(array([1, 2, 3, 4, 5, 6, 7, 8]));
        const pyResult = runNumPy(`
result = np.fft.rfft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float32', () => {
        const a = array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float32');
        const jsResult = fft.rfft(a);
        const pyResult = runNumPy(`
result = np.fft.rfft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float16', () => {
        const a = array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float16');
        const jsResult = fft.rfft(a);
        const pyResult = runNumPy(`
result = np.fft.rfft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float16))
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
      it('roundtrip matches NumPy', () => {
        const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
        const recovered = fft.irfft(fft.rfft(original));
        const pyResult = runNumPy(`
result = np.fft.irfft(np.fft.rfft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });

      it('roundtrip matches NumPy for float16', () => {
        const original = array([1, 2, 3, 4, 5, 6, 7, 8]).astype('float16');
        const recovered = fft.irfft(fft.rfft(original));
        const pyResult = runNumPy(`
result = np.fft.irfft(np.fft.rfft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float16)))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });

      it('roundtrip matches NumPy for N=1000', () => {
        const data = Array.from({ length: 1000 }, (_, i) => Math.sin(i * 0.1));
        const original = array(data);
        const recovered = fft.irfft(fft.rfft(original));
        const pyResult = runNumPy(`
result = np.fft.irfft(np.fft.rfft(np.sin(np.arange(1000) * 0.1)))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // fft2 / ifft2
    // ========================================

    describe('fft.fft2', () => {
      it('matches NumPy for float64', () => {
        const jsResult = fft.fft2(
          array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ]),
        );
        const pyResult = runNumPy(`
result = np.fft.fft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float16 2D', () => {
        const jsResult = fft.fft2(
          array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ]).astype('float16'),
        );
        const pyResult = runNumPy(`
result = np.fft.fft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for non-pow2 3x5', () => {
        const jsResult = fft.fft2(
          array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
          ]),
        );
        const pyResult = runNumPy(`
result = np.fft.fft2(np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]], dtype=float))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.ifft2', () => {
      it('roundtrip matches NumPy', () => {
        const original = array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]);
        const recovered = fft.ifft2(fft.fft2(original));
        const pyResult = runNumPy(`
result = np.fft.ifft2(np.fft.fft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]])))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // rfft2 / irfft2
    // ========================================

    describe('fft.rfft2', () => {
      it('matches NumPy for float64', () => {
        const jsResult = fft.rfft2(
          array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ]),
        );
        const pyResult = runNumPy(`
result = np.fft.rfft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.irfft2', () => {
      it('roundtrip matches NumPy', () => {
        const original = array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]);
        const recovered = fft.irfft2(fft.rfft2(original));
        const pyResult = runNumPy(`
result = np.fft.irfft2(np.fft.rfft2(np.array([[1, 2, 3, 4], [5, 6, 7, 8]])))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // fftn / ifftn
    // ========================================

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
          ]),
        );
        const pyResult = runNumPy(`
result = np.fft.fftn(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.ifftn', () => {
      it('roundtrip matches NumPy', () => {
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

    // ========================================
    // rfftn / irfftn
    // ========================================

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
          ]),
        );
        const pyResult = runNumPy(`
result = np.fft.rfftn(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float16', () => {
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
          ]).astype('float16'),
        );
        const pyResult = runNumPy(`
result = np.fft.rfftn(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.irfftn', () => {
      it('roundtrip matches NumPy', () => {
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

      it('roundtrip matches NumPy for float16', () => {
        const original = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]).astype('float16');
        const recovered = fft.irfftn(fft.rfftn(original));
        const pyResult = runNumPy(`
original = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
result = np.fft.irfftn(np.fft.rfftn(original))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // hfft / ihfft
    // ========================================

    describe('fft.hfft', () => {
      it('matches NumPy', () => {
        const jsResult = fft.hfft(
          array([new Complex(1, 0), new Complex(2, -1), new Complex(3, 0)]),
        );
        const pyResult = runNumPy(`
result = np.fft.hfft(np.array([1+0j, 2-1j, 3+0j]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.ihfft', () => {
      it('matches NumPy', () => {
        const jsResult = fft.ihfft(array([1, 2, 3, 4]));
        const pyResult = runNumPy(`
result = np.fft.ihfft(np.array([1, 2, 3, 4], dtype=float))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('roundtrip with hfft matches NumPy', () => {
        const original = array([1, 2, 3, 4, 5, 6, 7, 8]);
        const recovered = fft.hfft(fft.ihfft(original));
        const pyResult = runNumPy(`
result = np.fft.hfft(np.fft.ihfft(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // fftfreq / rfftfreq
    // ========================================

    describe('fft.fftfreq', () => {
      it('matches NumPy for even n', () => {
        const jsResult = fft.fftfreq(8);
        const pyResult = runNumPy(`result = np.fft.fftfreq(8)`);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for odd n', () => {
        const jsResult = fft.fftfreq(7);
        const pyResult = runNumPy(`result = np.fft.fftfreq(7)`);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy with custom spacing', () => {
        const jsResult = fft.fftfreq(8, 0.1);
        const pyResult = runNumPy(`result = np.fft.fftfreq(8, d=0.1)`);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.rfftfreq', () => {
      it('matches NumPy for even n', () => {
        const jsResult = fft.rfftfreq(8);
        const pyResult = runNumPy(`result = np.fft.rfftfreq(8)`);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for odd n', () => {
        const jsResult = fft.rfftfreq(7);
        const pyResult = runNumPy(`result = np.fft.rfftfreq(7)`);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // fftshift / ifftshift
    // ========================================

    describe('fft.fftshift', () => {
      it('matches NumPy for 1D', () => {
        const jsResult = fft.fftshift(array([0, 1, 2, 3, 4, -4, -3, -2, -1]));
        const pyResult = runNumPy(`
result = np.fft.fftshift(np.array([0, 1, 2, 3, 4, -4, -3, -2, -1]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2D', () => {
        const jsResult = fft.fftshift(
          array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
          ]),
        );
        const pyResult = runNumPy(`
result = np.fft.fftshift(np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]))
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('fft.ifftshift', () => {
      it('is inverse of fftshift (matches NumPy)', () => {
        const original = array([0, 1, 2, 3, 4, 5, 6, 7]);
        const recovered = fft.ifftshift(fft.fftshift(original));
        const pyResult = runNumPy(`
result = np.fft.ifftshift(np.fft.fftshift(np.array([0, 1, 2, 3, 4, 5, 6, 7])))
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });
    });

    // ========================================
    // Normalization modes
    // ========================================

    describe('normalization modes', () => {
      it('ortho roundtrip matches NumPy', () => {
        const original = array([1, 2, 3, 4, 5, 6]);
        const recovered = fft.ifft(
          fft.fft(original, undefined, undefined, 'ortho'),
          undefined,
          undefined,
          'ortho',
        );
        const pyResult = runNumPy(`
result = np.fft.ifft(np.fft.fft(np.array([1,2,3,4,5,6], dtype=float), norm='ortho'), norm='ortho')
        `);
        expect(arraysClose(recovered.toArray(), pyResult.value)).toBe(true);
      });

      it('forward normalization matches NumPy', () => {
        const jsResult = fft.fft(array([1, 2, 3, 4]), undefined, undefined, 'forward');
        const pyResult = runNumPy(`
result = np.fft.fft(np.array([1, 2, 3, 4]), norm='forward')
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });
  });
}
