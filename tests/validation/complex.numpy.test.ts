/**
 * Python NumPy validation tests for complex number support
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  zeros,
  ones,
  Complex,
  real,
  imag,
  conj,
  angle,
  abs,
  sqrt,
  power,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Complex Numbers', () => {
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

  describe('Array Creation', () => {
    it('creates zeros with complex128 dtype matching NumPy', () => {
      const jsResult = zeros([3], 'complex128');
      const pyResult = runNumPy(`
result = np.zeros(3, dtype=np.complex128)
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('creates ones with complex128 dtype matching NumPy', () => {
      const jsResult = ones([3], 'complex128');
      const pyResult = runNumPy(`
result = np.ones(3, dtype=np.complex128)
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('creates array from complex values matching NumPy', () => {
      const jsResult = array([new Complex(1, 2), new Complex(3, 4)]);
      const pyResult = runNumPy(`
result = np.array([1+2j, 3+4j])
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('real()', () => {
    it('extracts real part matching NumPy', () => {
      const jsResult = real(array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]));
      const pyResult = runNumPy(`
result = np.real(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(pyResult.dtype).toBe('float64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('imag()', () => {
    it('extracts imaginary part matching NumPy', () => {
      const jsResult = imag(array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]));
      const pyResult = runNumPy(`
result = np.imag(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(pyResult.dtype).toBe('float64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('conj()', () => {
    it('computes conjugate matching NumPy', () => {
      const jsResult = conj(array([new Complex(1, 2), new Complex(3, -4)]));
      const pyResult = runNumPy(`
result = np.conj(np.array([1+2j, 3-4j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('angle()', () => {
    it('computes phase angle matching NumPy', () => {
      const jsResult = angle(
        array([new Complex(1, 0), new Complex(0, 1), new Complex(-1, 0), new Complex(0, -1)])
      );
      const pyResult = runNumPy(`
result = np.angle(np.array([1+0j, 0+1j, -1+0j, 0-1j]))
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(pyResult.dtype).toBe('float64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes angle in degrees matching NumPy', () => {
      const jsResult = angle(array([new Complex(1, 0), new Complex(0, 1)]), true);
      const pyResult = runNumPy(`
result = np.angle(np.array([1+0j, 0+1j]), deg=True)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('abs()', () => {
    it('computes magnitude matching NumPy', () => {
      const jsResult = abs(array([new Complex(3, 4), new Complex(5, 12), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.abs(np.array([3+4j, 5+12j, 0+1j]))
      `);

      expect(jsResult.dtype).toBe('float64');
      expect(pyResult.dtype).toBe('float64');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Arithmetic', () => {
    it('adds complex arrays matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(5, 6), new Complex(7, 8)]);
      const jsResult = a.add(b);
      const pyResult = runNumPy(`
result = np.array([1+2j, 3+4j]) + np.array([5+6j, 7+8j])
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('subtracts complex arrays matching NumPy', () => {
      const a = array([new Complex(5, 6), new Complex(7, 8)]);
      const b = array([new Complex(1, 2), new Complex(3, 4)]);
      const jsResult = a.subtract(b);
      const pyResult = runNumPy(`
result = np.array([5+6j, 7+8j]) - np.array([1+2j, 3+4j])
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('multiplies complex arrays matching NumPy', () => {
      const a = array([new Complex(1, 2)]);
      const b = array([new Complex(3, 4)]);
      const jsResult = a.multiply(b);
      const pyResult = runNumPy(`
result = np.array([1+2j]) * np.array([3+4j])
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('divides complex arrays matching NumPy', () => {
      const a = array([new Complex(3, 4)]);
      const b = array([new Complex(1, 2)]);
      const jsResult = a.divide(b);
      const pyResult = runNumPy(`
result = np.array([3+4j]) / np.array([1+2j])
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('negates complex arrays matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(-3, 4)]);
      const jsResult = a.negative();
      const pyResult = runNumPy(`
result = -np.array([1+2j, -3+4j])
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('sqrt()', () => {
    it('computes sqrt of complex matching NumPy', () => {
      const jsResult = sqrt(array([new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.sqrt(np.array([0+1j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes sqrt of negative real as complex matching NumPy', () => {
      const jsResult = sqrt(array([new Complex(-1, 0)]));
      const pyResult = runNumPy(`
result = np.sqrt(np.array([-1+0j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('power()', () => {
    it('raises complex to real power matching NumPy', () => {
      const jsResult = power(array([new Complex(1, 1)]), 2);
      const pyResult = runNumPy(`
result = np.power(np.array([1+1j]), 2)
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('raises complex to fractional power matching NumPy', () => {
      const jsResult = power(array([new Complex(4, 0)]), 0.5);
      const pyResult = runNumPy(`
result = np.power(np.array([4+0j]), 0.5)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Type Promotion', () => {
    it('promotes complex + real to complex matching NumPy', () => {
      const a = array([new Complex(1, 2)]);
      const b = array([3]);
      const jsResult = a.add(b);
      const pyResult = runNumPy(`
result = np.array([1+2j]) + np.array([3])
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('promotes real + complex to complex matching NumPy', () => {
      const a = array([3]);
      const b = array([new Complex(1, 2)]);
      const jsResult = a.add(b);
      const pyResult = runNumPy(`
result = np.array([3]) + np.array([1+2j])
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Reductions', () => {
    it('sums complex array matching NumPy', () => {
      const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
      const jsResult = arr.sum();
      const pyResult = runNumPy(`
result = np.sum(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      const pyVal = pyResult.value;
      expect((jsResult as Complex).re).toBeCloseTo(pyVal.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyVal.im);
    });

    it('computes mean of complex array matching NumPy', () => {
      const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
      const jsResult = arr.mean();
      const pyResult = runNumPy(`
result = np.mean(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      const pyVal = pyResult.value;
      expect((jsResult as Complex).re).toBeCloseTo(pyVal.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyVal.im);
    });

    it('computes prod of complex array matching NumPy', () => {
      const arr = array([new Complex(1, 1), new Complex(2, 0)]);
      const jsResult = arr.prod();
      const pyResult = runNumPy(`
result = np.prod(np.array([1+1j, 2+0j]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      const pyVal = pyResult.value;
      expect((jsResult as Complex).re).toBeCloseTo(pyVal.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyVal.im);
    });

    it('sums complex array along axis matching NumPy', () => {
      const arr = array([
        [new Complex(1, 2), new Complex(3, 4)],
        [new Complex(5, 6), new Complex(7, 8)],
      ]);
      const jsResult = arr.sum(0);
      const pyResult = runNumPy(`
result = np.sum(np.array([[1+2j, 3+4j], [5+6j, 7+8j]]), axis=0)
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Comparisons', () => {
    it('compares complex arrays for equality matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(1, 2), new Complex(3, 5)]);
      const jsResult = a.equal(b);
      const pyResult = runNumPy(`
result = np.array([1+2j, 3+4j]) == np.array([1+2j, 3+5j])
      `);

      // Convert 0/1 to false/true for comparison
      const jsAsBool = jsResult.toArray().map((v: number) => v === 1);
      expect(jsAsBool).toEqual(pyResult.value);
    });

    it('compares complex arrays for inequality matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(1, 2), new Complex(3, 5)]);
      const jsResult = a.not_equal(b);
      const pyResult = runNumPy(`
result = np.array([1+2j, 3+4j]) != np.array([1+2j, 3+5j])
      `);

      // Convert 0/1 to false/true for comparison
      const jsAsBool = jsResult.toArray().map((v: number) => v === 1);
      expect(jsAsBool).toEqual(pyResult.value);
    });

    it('compares complex with scalar matching NumPy', () => {
      const a = array([new Complex(3, 0), new Complex(3, 1)]);
      const jsResult = a.equal(3);
      const pyResult = runNumPy(`
result = np.array([3+0j, 3+1j]) == 3
      `);

      // Convert 0/1 to false/true for comparison
      const jsAsBool = jsResult.toArray().map((v: number) => v === 1);
      expect(jsAsBool).toEqual(pyResult.value);
    });
  });
});
