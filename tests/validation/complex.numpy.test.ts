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
  exp,
  exp2,
  expm1,
  log,
  log2,
  log10,
  log1p,
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  sinh,
  cosh,
  tanh,
  arcsinh,
  arccosh,
  arctanh,
  positive,
  reciprocal,
  square,
  cumsum,
  cumprod,
  dot,
  trace,
  diagonal,
  transpose,
  inner,
  outer,
  kron,
  isfinite,
  isinf,
  isnan,
  nonzero,
  argwhere,
  flatnonzero,
  where,
  extract,
  count_nonzero,
  logical_and,
  logical_or,
  logical_not,
  logical_xor,
  diff,
  ediff1d,
  gradient,
  cross,
  nancumsum,
  nancumprod,
  nanvar,
  nanstd,
  nanargmin,
  nanargmax,
  average,
  sort,
  argsort,
  max,
  min,
  nanmin,
  nanmax,
  ptp,
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

  describe('exp()', () => {
    it('computes exp of purely imaginary number matching NumPy', () => {
      // exp(i*pi) = -1 (Euler's identity)
      const jsResult = exp(array([new Complex(0, Math.PI)]));
      const pyResult = runNumPy(`
import cmath
result = np.exp(np.array([0+1j*cmath.pi]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes exp of real number matching NumPy', () => {
      const jsResult = exp(array([new Complex(1, 0), new Complex(0, 0), new Complex(-1, 0)]));
      const pyResult = runNumPy(`
result = np.exp(np.array([1+0j, 0+0j, -1+0j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes exp of complex number matching NumPy', () => {
      const jsResult = exp(array([new Complex(1, 1), new Complex(2, 0.5), new Complex(-1, 2)]));
      const pyResult = runNumPy(`
result = np.exp(np.array([1+1j, 2+0.5j, -1+2j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log()', () => {
    it('computes log of positive real matching NumPy', () => {
      const jsResult = log(array([new Complex(1, 0), new Complex(Math.E, 0), new Complex(10, 0)]));
      const pyResult = runNumPy(`
import math
result = np.log(np.array([1+0j, math.e+0j, 10+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes log of negative real matching NumPy', () => {
      const jsResult = log(array([new Complex(-1, 0), new Complex(-Math.E, 0)]));
      const pyResult = runNumPy(`
import math
result = np.log(np.array([-1+0j, -math.e+0j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes log of complex number matching NumPy', () => {
      const jsResult = log(array([new Complex(1, 1), new Complex(3, 4), new Complex(-1, 1)]));
      const pyResult = runNumPy(`
result = np.log(np.array([1+1j, 3+4j, -1+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('exp2()', () => {
    it('computes 2^z matching NumPy', () => {
      const jsResult = exp2(array([new Complex(0, 0), new Complex(1, 0), new Complex(3, 0)]));
      const pyResult = runNumPy(`
result = np.exp2(np.array([0+0j, 1+0j, 3+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes 2^(complex) matching NumPy', () => {
      const jsResult = exp2(array([new Complex(1, 1), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.exp2(np.array([1+1j, 0+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('expm1()', () => {
    it('computes exp(z)-1 matching NumPy', () => {
      const jsResult = expm1(array([new Complex(0, 0), new Complex(1, 0)]));
      const pyResult = runNumPy(`
result = np.expm1(np.array([0+0j, 1+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes expm1 of complex matching NumPy', () => {
      const jsResult = expm1(array([new Complex(0, Math.PI), new Complex(1, 0.5)]));
      const pyResult = runNumPy(`
import cmath
result = np.expm1(np.array([0+1j*cmath.pi, 1+0.5j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log2()', () => {
    it('computes log2 of positive real matching NumPy', () => {
      const jsResult = log2(array([new Complex(1, 0), new Complex(2, 0), new Complex(8, 0)]));
      const pyResult = runNumPy(`
result = np.log2(np.array([1+0j, 2+0j, 8+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes log2 of complex matching NumPy', () => {
      const jsResult = log2(array([new Complex(-1, 0), new Complex(1, 1)]));
      const pyResult = runNumPy(`
result = np.log2(np.array([-1+0j, 1+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log10()', () => {
    it('computes log10 of positive real matching NumPy', () => {
      const jsResult = log10(array([new Complex(1, 0), new Complex(10, 0), new Complex(100, 0)]));
      const pyResult = runNumPy(`
result = np.log10(np.array([1+0j, 10+0j, 100+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes log10 of complex matching NumPy', () => {
      const jsResult = log10(array([new Complex(-1, 0), new Complex(3, 4)]));
      const pyResult = runNumPy(`
result = np.log10(np.array([-1+0j, 3+4j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('log1p()', () => {
    it('computes log(1+z) matching NumPy', () => {
      const jsResult = log1p(array([new Complex(0, 0), new Complex(Math.E - 1, 0)]));
      const pyResult = runNumPy(`
import math
result = np.log1p(np.array([0+0j, (math.e-1)+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes log1p of complex matching NumPy', () => {
      const jsResult = log1p(array([new Complex(0, 1), new Complex(-2, 0), new Complex(1, 1)]));
      const pyResult = runNumPy(`
result = np.log1p(np.array([0+1j, -2+0j, 1+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('exp and log inverse relationship', () => {
    it('exp(log(z)) = z matching NumPy', () => {
      const z = array([new Complex(3, 4), new Complex(1, 2), new Complex(5, -3)]);
      const jsResult = exp(log(z));
      const pyResult = runNumPy(`
result = np.exp(np.log(np.array([3+4j, 1+2j, 5-3j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('log(exp(z)) = z for principal branch matching NumPy', () => {
      // For principal branch, this works when -pi < im(z) <= pi
      const z = array([new Complex(1, 0.5), new Complex(2, -1), new Complex(0.5, 1)]);
      const jsResult = log(exp(z));
      const pyResult = runNumPy(`
result = np.log(np.exp(np.array([1+0.5j, 2-1j, 0.5+1j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  // ==========================================================================
  // Trigonometric Functions
  // ==========================================================================

  describe('sin()', () => {
    it('computes sin of real values matching NumPy', () => {
      const jsResult = sin(array([new Complex(0, 0), new Complex(Math.PI / 2, 0), new Complex(Math.PI, 0)]));
      const pyResult = runNumPy(`
import math
result = np.sin(np.array([0+0j, math.pi/2+0j, math.pi+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes sin of complex matching NumPy', () => {
      const jsResult = sin(array([new Complex(0, 1), new Complex(1, 1), new Complex(0.5, 2)]));
      const pyResult = runNumPy(`
result = np.sin(np.array([0+1j, 1+1j, 0.5+2j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cos()', () => {
    it('computes cos of real values matching NumPy', () => {
      const jsResult = cos(array([new Complex(0, 0), new Complex(Math.PI / 2, 0), new Complex(Math.PI, 0)]));
      const pyResult = runNumPy(`
import math
result = np.cos(np.array([0+0j, math.pi/2+0j, math.pi+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes cos of complex matching NumPy', () => {
      const jsResult = cos(array([new Complex(0, 1), new Complex(1, 1), new Complex(0.5, 2)]));
      const pyResult = runNumPy(`
result = np.cos(np.array([0+1j, 1+1j, 0.5+2j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tan()', () => {
    it('computes tan of real values matching NumPy', () => {
      const jsResult = tan(array([new Complex(0, 0), new Complex(Math.PI / 4, 0), new Complex(1, 0)]));
      const pyResult = runNumPy(`
import math
result = np.tan(np.array([0+0j, math.pi/4+0j, 1+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes tan of complex matching NumPy', () => {
      const jsResult = tan(array([new Complex(0, 1), new Complex(0.5, 0.5), new Complex(1, 0.3)]));
      const pyResult = runNumPy(`
result = np.tan(np.array([0+1j, 0.5+0.5j, 1+0.3j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arcsin()', () => {
    it('computes arcsin of real values matching NumPy', () => {
      const jsResult = arcsin(array([new Complex(0, 0), new Complex(0.5, 0), new Complex(1, 0)]));
      const pyResult = runNumPy(`
result = np.arcsin(np.array([0+0j, 0.5+0j, 1+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes arcsin of complex matching NumPy', () => {
      const jsResult = arcsin(array([new Complex(0.3, 0.4), new Complex(2, 0), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.arcsin(np.array([0.3+0.4j, 2+0j, 0+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arccos()', () => {
    it('computes arccos of real values matching NumPy', () => {
      const jsResult = arccos(array([new Complex(0, 0), new Complex(0.5, 0), new Complex(1, 0)]));
      const pyResult = runNumPy(`
result = np.arccos(np.array([0+0j, 0.5+0j, 1+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes arccos of complex matching NumPy', () => {
      const jsResult = arccos(array([new Complex(0.3, 0.4), new Complex(2, 0), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.arccos(np.array([0.3+0.4j, 2+0j, 0+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arctan()', () => {
    it('computes arctan of real values matching NumPy', () => {
      const jsResult = arctan(array([new Complex(0, 0), new Complex(1, 0), new Complex(2, 0)]));
      const pyResult = runNumPy(`
result = np.arctan(np.array([0+0j, 1+0j, 2+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes arctan of complex matching NumPy', () => {
      const jsResult = arctan(array([new Complex(0.3, 0.2), new Complex(1, 0.5), new Complex(0, 0.5)]));
      const pyResult = runNumPy(`
result = np.arctan(np.array([0.3+0.2j, 1+0.5j, 0+0.5j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('sin/arcsin inverse relationship', () => {
    it('arcsin(sin(z)) = z for small z matching NumPy', () => {
      const z = array([new Complex(0.3, 0.4), new Complex(0.5, 0.5), new Complex(0.2, 0.1)]);
      const jsResult = arcsin(sin(z));
      const pyResult = runNumPy(`
result = np.arcsin(np.sin(np.array([0.3+0.4j, 0.5+0.5j, 0.2+0.1j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('sin(arcsin(z)) = z matching NumPy', () => {
      const z = array([new Complex(0.7, 0.3), new Complex(0.5, 0.2), new Complex(0.3, 0.1)]);
      const jsResult = sin(arcsin(z));
      const pyResult = runNumPy(`
result = np.sin(np.arcsin(np.array([0.7+0.3j, 0.5+0.2j, 0.3+0.1j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cos/arccos inverse relationship', () => {
    it('arccos(cos(z)) = z for z in principal range matching NumPy', () => {
      const z = array([new Complex(0.5, 0.3), new Complex(1.0, 0.5), new Complex(0.8, 0.2)]);
      const jsResult = arccos(cos(z));
      const pyResult = runNumPy(`
result = np.arccos(np.cos(np.array([0.5+0.3j, 1.0+0.5j, 0.8+0.2j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('cos(arccos(z)) = z matching NumPy', () => {
      const z = array([new Complex(0.5, 0.5), new Complex(0.3, 0.2), new Complex(0.7, 0.3)]);
      const jsResult = cos(arccos(z));
      const pyResult = runNumPy(`
result = np.cos(np.arccos(np.array([0.5+0.5j, 0.3+0.2j, 0.7+0.3j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tan/arctan inverse relationship', () => {
    it('arctan(tan(z)) = z for small z matching NumPy', () => {
      const z = array([new Complex(0.3, 0.2), new Complex(0.5, 0.3), new Complex(0.2, 0.1)]);
      const jsResult = arctan(tan(z));
      const pyResult = runNumPy(`
result = np.arctan(np.tan(np.array([0.3+0.2j, 0.5+0.3j, 0.2+0.1j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('tan(arctan(z)) = z matching NumPy', () => {
      const z = array([new Complex(1.5, 0.5), new Complex(0.8, 0.3), new Complex(2, 0.5)]);
      const jsResult = tan(arctan(z));
      const pyResult = runNumPy(`
result = np.tan(np.arctan(np.array([1.5+0.5j, 0.8+0.3j, 2+0.5j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Pythagorean identity', () => {
    it('sin²(z) + cos²(z) = 1 matching NumPy', () => {
      const z = array([new Complex(1.2, 0.8), new Complex(0.5, 1.5), new Complex(2, 0.3)]);
      const sinResult = sin(z);
      const cosResult = cos(z);
      const pyResult = runNumPy(`
z = np.array([1.2+0.8j, 0.5+1.5j, 2+0.3j])
sin_result = np.sin(z)
cos_result = np.cos(z)
result = sin_result**2 + cos_result**2
      `);

      // Compute sin²(z) + cos²(z) in JS
      const sinVals = sinResult.toArray();
      const cosVals = cosResult.toArray();
      const sumVals = sinVals.map((s, i) => {
        const c = cosVals[i];
        // (a+bi)² = a² - b² + 2abi
        const sin2Re = s.re * s.re - s.im * s.im;
        const sin2Im = 2 * s.re * s.im;
        const cos2Re = c.re * c.re - c.im * c.im;
        const cos2Im = 2 * c.re * c.im;
        return new Complex(sin2Re + cos2Re, sin2Im + cos2Im);
      });

      expect(arraysClose(sumVals, pyResult.value)).toBe(true);
    });
  });

  // ==========================================================================
  // Hyperbolic Functions
  // ==========================================================================

  describe('sinh()', () => {
    it('computes sinh of real values matching NumPy', () => {
      const jsResult = sinh(array([new Complex(0, 0), new Complex(1, 0), new Complex(2, 0)]));
      const pyResult = runNumPy(`
result = np.sinh(np.array([0+0j, 1+0j, 2+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes sinh of complex matching NumPy', () => {
      const jsResult = sinh(array([new Complex(0, 1), new Complex(1, 1), new Complex(0.5, 2)]));
      const pyResult = runNumPy(`
result = np.sinh(np.array([0+1j, 1+1j, 0.5+2j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cosh()', () => {
    it('computes cosh of real values matching NumPy', () => {
      const jsResult = cosh(array([new Complex(0, 0), new Complex(1, 0), new Complex(2, 0)]));
      const pyResult = runNumPy(`
result = np.cosh(np.array([0+0j, 1+0j, 2+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes cosh of complex matching NumPy', () => {
      const jsResult = cosh(array([new Complex(0, 1), new Complex(1, 1), new Complex(0.5, 2)]));
      const pyResult = runNumPy(`
result = np.cosh(np.array([0+1j, 1+1j, 0.5+2j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tanh()', () => {
    it('computes tanh of real values matching NumPy', () => {
      const jsResult = tanh(array([new Complex(0, 0), new Complex(1, 0), new Complex(0.5, 0)]));
      const pyResult = runNumPy(`
result = np.tanh(np.array([0+0j, 1+0j, 0.5+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes tanh of complex matching NumPy', () => {
      const jsResult = tanh(array([new Complex(0, 1), new Complex(0.5, 0.5), new Complex(1, 0.3)]));
      const pyResult = runNumPy(`
result = np.tanh(np.array([0+1j, 0.5+0.5j, 1+0.3j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arcsinh()', () => {
    it('computes arcsinh of real values matching NumPy', () => {
      const jsResult = arcsinh(array([new Complex(0, 0), new Complex(1, 0), new Complex(2, 0)]));
      const pyResult = runNumPy(`
result = np.arcsinh(np.array([0+0j, 1+0j, 2+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes arcsinh of complex matching NumPy', () => {
      const jsResult = arcsinh(array([new Complex(0.3, 0.4), new Complex(1, 1), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.arcsinh(np.array([0.3+0.4j, 1+1j, 0+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arccosh()', () => {
    it('computes arccosh of real values >= 1 matching NumPy', () => {
      const jsResult = arccosh(array([new Complex(1, 0), new Complex(2, 0), new Complex(3, 0)]));
      const pyResult = runNumPy(`
result = np.arccosh(np.array([1+0j, 2+0j, 3+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes arccosh of complex matching NumPy', () => {
      const jsResult = arccosh(array([new Complex(0.5, 0.3), new Complex(2, 0.5), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.arccosh(np.array([0.5+0.3j, 2+0.5j, 0+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arctanh()', () => {
    it('computes arctanh of real values in (-1, 1) matching NumPy', () => {
      const jsResult = arctanh(array([new Complex(0, 0), new Complex(0.5, 0), new Complex(-0.5, 0)]));
      const pyResult = runNumPy(`
result = np.arctanh(np.array([0+0j, 0.5+0j, -0.5+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes arctanh of complex matching NumPy', () => {
      const jsResult = arctanh(array([new Complex(0.3, 0.2), new Complex(0.5, 0.5), new Complex(0, 0.5)]));
      const pyResult = runNumPy(`
result = np.arctanh(np.array([0.3+0.2j, 0.5+0.5j, 0+0.5j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('sinh/arcsinh inverse relationship', () => {
    it('arcsinh(sinh(z)) = z matching NumPy', () => {
      const z = array([new Complex(0.3, 0.4), new Complex(0.5, 0.5), new Complex(0.2, 0.1)]);
      const jsResult = arcsinh(sinh(z));
      const pyResult = runNumPy(`
result = np.arcsinh(np.sinh(np.array([0.3+0.4j, 0.5+0.5j, 0.2+0.1j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('sinh(arcsinh(z)) = z matching NumPy', () => {
      const z = array([new Complex(1.5, 0.5), new Complex(0.8, 0.3), new Complex(2, 0.5)]);
      const jsResult = sinh(arcsinh(z));
      const pyResult = runNumPy(`
result = np.sinh(np.arcsinh(np.array([1.5+0.5j, 0.8+0.3j, 2+0.5j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cosh/arccosh inverse relationship', () => {
    it('cosh(arccosh(z)) = z matching NumPy', () => {
      const z = array([new Complex(2, 0.5), new Complex(1.5, 0.3), new Complex(3, 0.2)]);
      const jsResult = cosh(arccosh(z));
      const pyResult = runNumPy(`
result = np.cosh(np.arccosh(np.array([2+0.5j, 1.5+0.3j, 3+0.2j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tanh/arctanh inverse relationship', () => {
    it('arctanh(tanh(z)) = z for small z matching NumPy', () => {
      const z = array([new Complex(0.3, 0.2), new Complex(0.5, 0.3), new Complex(0.2, 0.1)]);
      const jsResult = arctanh(tanh(z));
      const pyResult = runNumPy(`
result = np.arctanh(np.tanh(np.array([0.3+0.2j, 0.5+0.3j, 0.2+0.1j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('tanh(arctanh(z)) = z matching NumPy', () => {
      const z = array([new Complex(0.5, 0.5), new Complex(0.3, 0.2), new Complex(0.7, 0.3)]);
      const jsResult = tanh(arctanh(z));
      const pyResult = runNumPy(`
result = np.tanh(np.arctanh(np.array([0.5+0.5j, 0.3+0.2j, 0.7+0.3j])))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Hyperbolic identity', () => {
    it('cosh²(z) - sinh²(z) = 1 matching NumPy', () => {
      const z = array([new Complex(1.2, 0.8), new Complex(0.5, 1.5), new Complex(2, 0.3)]);
      const sinhResult = sinh(z);
      const coshResult = cosh(z);
      const pyResult = runNumPy(`
z = np.array([1.2+0.8j, 0.5+1.5j, 2+0.3j])
sinh_result = np.sinh(z)
cosh_result = np.cosh(z)
result = cosh_result**2 - sinh_result**2
      `);

      // Compute cosh²(z) - sinh²(z) in JS
      const sinhVals = sinhResult.toArray();
      const coshVals = coshResult.toArray();
      const diffVals = sinhVals.map((s, i) => {
        const c = coshVals[i];
        const sinh2Re = s.re * s.re - s.im * s.im;
        const sinh2Im = 2 * s.re * s.im;
        const cosh2Re = c.re * c.re - c.im * c.im;
        const cosh2Im = 2 * c.re * c.im;
        return new Complex(cosh2Re - sinh2Re, cosh2Im - sinh2Im);
      });

      expect(arraysClose(diffVals, pyResult.value)).toBe(true);
    });
  });

  // ==========================================================================
  // Arithmetic Functions
  // ==========================================================================

  describe('positive()', () => {
    it('returns copy of complex array matching NumPy', () => {
      const jsResult = positive(array([new Complex(3, 4), new Complex(-1, 2), new Complex(0, 0)]));
      const pyResult = runNumPy(`
result = np.positive(np.array([3+4j, -1+2j, 0+0j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('preserves negative complex values matching NumPy', () => {
      const jsResult = positive(array([new Complex(-5, -3), new Complex(-2, 4)]));
      const pyResult = runNumPy(`
result = np.positive(np.array([-5-3j, -2+4j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('reciprocal()', () => {
    it('computes 1/z matching NumPy', () => {
      const jsResult = reciprocal(array([new Complex(1, 0), new Complex(3, 4), new Complex(0, 1)]));
      const pyResult = runNumPy(`
result = np.reciprocal(np.array([1+0j, 3+4j, 0+1j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes reciprocal of various complex values matching NumPy', () => {
      const jsResult = reciprocal(array([new Complex(2, 3), new Complex(0, 2), new Complex(1, 1)]));
      const pyResult = runNumPy(`
result = np.reciprocal(np.array([2+3j, 0+2j, 1+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('square()', () => {
    it('computes z² matching NumPy', () => {
      const jsResult = square(array([new Complex(1, 0), new Complex(0, 1), new Complex(3, 4)]));
      const pyResult = runNumPy(`
result = np.square(np.array([1+0j, 0+1j, 3+4j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes square of various complex values matching NumPy', () => {
      const jsResult = square(array([new Complex(1, 1), new Complex(-2, -3), new Complex(0, 0)]));
      const pyResult = runNumPy(`
result = np.square(np.array([1+1j, -2-3j, 0+0j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  // ==========================================================================
  // Reduction Functions
  // ==========================================================================

  describe('sum()', () => {
    it('computes sum of complex array matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
      const jsResult = a.sum();
      const pyResult = runNumPy(`
result = np.sum(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });
  });

  describe('prod()', () => {
    it('computes product of complex array matching NumPy', () => {
      const a = array([new Complex(1, 1), new Complex(2, 1), new Complex(1, 0)]);
      const jsResult = a.prod();
      const pyResult = runNumPy(`
result = np.prod(np.array([1+1j, 2+1j, 1+0j]))
      `);

      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re, 5);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im, 5);
    });
  });

  describe('mean()', () => {
    it('computes mean of complex array matching NumPy', () => {
      const a = array([new Complex(2, 4), new Complex(4, 8), new Complex(6, 12)]);
      const jsResult = a.mean();
      const pyResult = runNumPy(`
result = np.mean(np.array([2+4j, 4+8j, 6+12j]))
      `);

      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });
  });

  describe('var()', () => {
    it('computes variance of complex array matching NumPy', () => {
      const a = array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3)]);
      const jsResult = a.var();
      const pyResult = runNumPy(`
result = np.var(np.array([1+1j, 2+2j, 3+3j]))
      `);

      expect(typeof jsResult).toBe('number');
      expect(jsResult as number).toBeCloseTo(pyResult.value as number);
    });

    it('computes variance with complex values matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 1), new Complex(2, 4)]);
      const jsResult = a.var();
      const pyResult = runNumPy(`
result = np.var(np.array([1+2j, 3+1j, 2+4j]))
      `);

      expect(jsResult as number).toBeCloseTo(pyResult.value as number);
    });
  });

  describe('std()', () => {
    it('computes std of complex array matching NumPy', () => {
      const a = array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3)]);
      const jsResult = a.std();
      const pyResult = runNumPy(`
result = np.std(np.array([1+1j, 2+2j, 3+3j]))
      `);

      expect(typeof jsResult).toBe('number');
      expect(jsResult as number).toBeCloseTo(pyResult.value as number);
    });
  });

  describe('cumsum()', () => {
    it('computes cumulative sum of complex array matching NumPy', () => {
      const jsResult = cumsum(array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]));
      const pyResult = runNumPy(`
result = np.cumsum(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes cumsum with various complex values matching NumPy', () => {
      const jsResult = cumsum(array([new Complex(0, 1), new Complex(1, 0), new Complex(-1, -1)]));
      const pyResult = runNumPy(`
result = np.cumsum(np.array([0+1j, 1+0j, -1-1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cumprod()', () => {
    it('computes cumulative product of complex array matching NumPy', () => {
      const jsResult = cumprod(array([new Complex(1, 1), new Complex(2, 0), new Complex(1, 1)]));
      const pyResult = runNumPy(`
result = np.cumprod(np.array([1+1j, 2+0j, 1+1j]))
      `);

      expect(jsResult.dtype).toBe('complex128');
      expect(pyResult.dtype).toBe('complex128');
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes cumprod with various complex values matching NumPy', () => {
      const jsResult = cumprod(array([new Complex(1, 0), new Complex(0, 1), new Complex(2, 1)]));
      const pyResult = runNumPy(`
result = np.cumprod(np.array([1+0j, 0+1j, 2+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('dot()', () => {
    it('computes dot product of complex 1D arrays matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(5, 6), new Complex(7, 8)]);
      const jsResult = dot(a, b);
      const pyResult = runNumPy(`
result = np.dot(np.array([1+2j, 3+4j]), np.array([5+6j, 7+8j]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });

    it('computes matrix-vector dot product matching NumPy', () => {
      const A = array([
        [new Complex(1, 0), new Complex(0, 1)],
        [new Complex(0, -1), new Complex(1, 0)],
      ]);
      const v = array([new Complex(2, 1), new Complex(1, 2)]);
      const jsResult = dot(A, v);
      const pyResult = runNumPy(`
A = np.array([[1+0j, 0+1j], [0-1j, 1+0j]])
v = np.array([2+1j, 1+2j])
result = np.dot(A, v)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes matrix multiplication via dot matching NumPy', () => {
      const A = array([
        [new Complex(1, 1), new Complex(2, 0)],
        [new Complex(0, 1), new Complex(1, -1)],
      ]);
      const B = array([
        [new Complex(1, 0), new Complex(0, 1)],
        [new Complex(2, 1), new Complex(1, 0)],
      ]);
      const jsResult = dot(A, B);
      const pyResult = runNumPy(`
A = np.array([[1+1j, 2+0j], [0+1j, 1-1j]])
B = np.array([[1+0j, 0+1j], [2+1j, 1+0j]])
result = np.dot(A, B)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('trace()', () => {
    it('computes trace of complex matrix matching NumPy', () => {
      const A = array([
        [new Complex(1, 2), new Complex(3, 4)],
        [new Complex(5, 6), new Complex(7, 8)],
      ]);
      const jsResult = trace(A);
      const pyResult = runNumPy(`
result = np.trace(np.array([[1+2j, 3+4j], [5+6j, 7+8j]]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });

    it('computes trace of larger complex matrix matching NumPy', () => {
      const A = array([
        [new Complex(1, 0), new Complex(2, 0), new Complex(3, 0)],
        [new Complex(4, 0), new Complex(5, 1), new Complex(6, 0)],
        [new Complex(7, 0), new Complex(8, 0), new Complex(9, -1)],
      ]);
      const jsResult = trace(A);
      const pyResult = runNumPy(`
result = np.trace(np.array([[1+0j, 2+0j, 3+0j], [4+0j, 5+1j, 6+0j], [7+0j, 8+0j, 9-1j]]))
      `);

      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });
  });

  describe('diagonal()', () => {
    it('extracts diagonal from complex matrix matching NumPy', () => {
      const A = array([
        [new Complex(1, 2), new Complex(3, 4)],
        [new Complex(5, 6), new Complex(7, 8)],
      ]);
      const jsResult = diagonal(A);
      const pyResult = runNumPy(`
result = np.diagonal(np.array([[1+2j, 3+4j], [5+6j, 7+8j]]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('extracts superdiagonal from complex matrix matching NumPy', () => {
      const A = array([
        [new Complex(1, 0), new Complex(2, 1), new Complex(3, 0)],
        [new Complex(4, 0), new Complex(5, 0), new Complex(6, 1)],
        [new Complex(7, 0), new Complex(8, 0), new Complex(9, 0)],
      ]);
      const jsResult = diagonal(A, 1);
      const pyResult = runNumPy(`
result = np.diagonal(np.array([[1+0j, 2+1j, 3+0j], [4+0j, 5+0j, 6+1j], [7+0j, 8+0j, 9+0j]]), offset=1)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('transpose()', () => {
    it('transposes complex matrix matching NumPy', () => {
      const A = array([
        [new Complex(1, 2), new Complex(3, 4)],
        [new Complex(5, 6), new Complex(7, 8)],
      ]);
      const jsResult = transpose(A);
      const pyResult = runNumPy(`
result = np.transpose(np.array([[1+2j, 3+4j], [5+6j, 7+8j]]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('inner()', () => {
    it('computes inner product of complex 1D arrays matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4)]);
      const b = array([new Complex(5, 6), new Complex(7, 8)]);
      const jsResult = inner(a, b);
      const pyResult = runNumPy(`
result = np.inner(np.array([1+2j, 3+4j]), np.array([5+6j, 7+8j]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });
  });

  describe('outer()', () => {
    it('computes outer product of complex arrays matching NumPy', () => {
      const a = array([new Complex(1, 0), new Complex(0, 1)]);
      const b = array([new Complex(1, 0), new Complex(0, 1)]);
      const jsResult = outer(a, b);
      const pyResult = runNumPy(`
result = np.outer(np.array([1+0j, 0+1j]), np.array([1+0j, 0+1j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes outer product of larger complex arrays matching NumPy', () => {
      const a = array([new Complex(1, 1), new Complex(2, -1), new Complex(0, 3)]);
      const b = array([new Complex(2, 0), new Complex(0, 2)]);
      const jsResult = outer(a, b);
      const pyResult = runNumPy(`
result = np.outer(np.array([1+1j, 2-1j, 0+3j]), np.array([2+0j, 0+2j]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('kron()', () => {
    it('computes Kronecker product of complex matrices matching NumPy', () => {
      const A = array([[new Complex(1, 0), new Complex(0, 1)]]);
      const B = array([[new Complex(1, 0)], [new Complex(0, 1)]]);
      const jsResult = kron(A, B);
      const pyResult = runNumPy(`
result = np.kron(np.array([[1+0j, 0+1j]]), np.array([[1+0j], [0+1j]]))
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('computes Kronecker product of 2x2 complex matrices matching NumPy', () => {
      const A = array([
        [new Complex(1, 1), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(1, -1)],
      ]);
      const B = array([
        [new Complex(1, 0), new Complex(0, 1)],
        [new Complex(0, -1), new Complex(1, 0)],
      ]);
      const jsResult = kron(A, B);
      const pyResult = runNumPy(`
A = np.array([[1+1j, 0+0j], [0+0j, 1-1j]])
B = np.array([[1+0j, 0+1j], [0-1j, 1+0j]])
result = np.kron(A, B)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('Logic/Checking Operations', () => {
    describe('isfinite()', () => {
      it('checks finite for complex numbers matching NumPy', () => {
        const a = array([
          new Complex(1, 2),
          new Complex(Infinity, 0),
          new Complex(0, Infinity),
          new Complex(NaN, 0),
        ]);
        const jsResult = isfinite(a);
        const pyResult = runNumPy(`
result = np.isfinite(np.array([1+2j, np.inf+0j, 0+np.inf*1j, np.nan+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });

    describe('isinf()', () => {
      it('checks infinity for complex numbers matching NumPy', () => {
        const a = array([
          new Complex(1, 2),
          new Complex(Infinity, 0),
          new Complex(0, -Infinity),
          new Complex(NaN, 0),
        ]);
        const jsResult = isinf(a);
        const pyResult = runNumPy(`
result = np.isinf(np.array([1+2j, np.inf+0j, 0-np.inf*1j, np.nan+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });

    describe('isnan()', () => {
      it('checks NaN for complex numbers matching NumPy', () => {
        const a = array([
          new Complex(1, 2),
          new Complex(NaN, 0),
          new Complex(0, NaN),
          new Complex(Infinity, 0),
        ]);
        const jsResult = isnan(a);
        const pyResult = runNumPy(`
result = np.isnan(np.array([1+2j, np.nan+0j, 0+np.nan*1j, np.inf+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });

    // Note: NumPy doesn't support isneginf/isposinf for complex types
    // (considers them ambiguous), so we skip validation tests for those.
    // Our implementation checks if either part is ±infinity.
  });

  describe('Searching/Indexing Operations', () => {
    describe('nonzero()', () => {
      it('finds nonzero indices in complex array matching NumPy', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const jsResult = nonzero(a);
        const pyResult = runNumPy(`
result = [arr.tolist() for arr in np.nonzero(np.array([0+0j, 1+0j, 0+1j, 0+0j]))]
        `);

        expect(jsResult[0].toArray()).toEqual(pyResult.value[0]);
      });
    });

    describe('argwhere()', () => {
      it('finds argwhere indices in complex array matching NumPy', () => {
        const a = array([new Complex(0, 0), new Complex(1, 2), new Complex(0, 0), new Complex(3, 0)]);
        const jsResult = argwhere(a);
        const pyResult = runNumPy(`
result = np.argwhere(np.array([0+0j, 1+2j, 0+0j, 3+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('flatnonzero()', () => {
      it('finds flatnonzero indices in complex array matching NumPy', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const jsResult = flatnonzero(a);
        const pyResult = runNumPy(`
result = np.flatnonzero(np.array([0+0j, 1+0j, 0+1j, 0+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('where()', () => {
      it('selects from complex arrays based on condition matching NumPy', () => {
        const cond = array([1, 0, 1]);
        const x = array([new Complex(10, 10), new Complex(20, 20), new Complex(30, 30)]);
        const y = array([new Complex(100, 100), new Complex(200, 200), new Complex(300, 300)]);
        const jsResult = where(cond, x, y);
        const pyResult = runNumPy(`
result = np.where(np.array([1, 0, 1]), np.array([10+10j, 20+20j, 30+30j]), np.array([100+100j, 200+200j, 300+300j]))
        `);

        expect(arraysClose((jsResult as ReturnType<typeof array>).toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('extract()', () => {
      it('extracts complex values matching NumPy', () => {
        const cond = array([1, 0, 1, 0]);
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8)]);
        const jsResult = extract(cond, a);
        const pyResult = runNumPy(`
result = np.extract(np.array([1, 0, 1, 0]), np.array([1+2j, 3+4j, 5+6j, 7+8j]))
        `);

        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('count_nonzero()', () => {
      it('counts nonzero complex values matching NumPy', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const jsResult = count_nonzero(a);
        const pyResult = runNumPy(`
result = np.count_nonzero(np.array([0+0j, 1+0j, 0+1j, 0+0j]))
        `);

        expect(jsResult).toBe(pyResult.value);
      });
    });
  });

  describe('Logical Operations', () => {
    describe('logical_and()', () => {
      it('computes logical AND of complex arrays matching NumPy', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const b = array([new Complex(1, 0), new Complex(1, 0), new Complex(1, 0), new Complex(0, 0)]);
        const jsResult = logical_and(a, b);
        const pyResult = runNumPy(`
result = np.logical_and(np.array([1+0j, 0+0j, 0+1j, 0+0j]), np.array([1+0j, 1+0j, 1+0j, 0+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });

    describe('logical_or()', () => {
      it('computes logical OR of complex arrays matching NumPy', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const b = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)]);
        const jsResult = logical_or(a, b);
        const pyResult = runNumPy(`
result = np.logical_or(np.array([1+0j, 0+0j, 0+1j, 0+0j]), np.array([0+0j, 1+0j, 0+0j, 0+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });

    describe('logical_not()', () => {
      it('computes logical NOT of complex array matching NumPy', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const jsResult = logical_not(a);
        const pyResult = runNumPy(`
result = np.logical_not(np.array([1+0j, 0+0j, 0+1j, 0+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });

    describe('logical_xor()', () => {
      it('computes logical XOR of complex arrays matching NumPy', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const b = array([new Complex(1, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)]);
        const jsResult = logical_xor(a, b);
        const pyResult = runNumPy(`
result = np.logical_xor(np.array([1+0j, 0+0j, 0+1j, 0+0j]), np.array([1+0j, 1+0j, 0+0j, 0+0j]))
        `);

        expect(jsResult.toArray()).toEqual(pyResult.value.map((v: boolean) => (v ? 1 : 0)));
      });
    });
  });

  // ==========================================================================
  // Difference Operations
  // ==========================================================================

  describe('Difference Operations', () => {
    describe('diff()', () => {
      it('computes 1D diff of complex array matching NumPy', () => {
        const a = array([new Complex(1, 2), new Complex(4, 3), new Complex(6, 1), new Complex(7, 5)]);
        const jsResult = diff(a);
        const pyResult = runNumPy(`
result = np.diff(np.array([1+2j, 4+3j, 6+1j, 7+5j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('computes diff with n=2 matching NumPy', () => {
        const a = array([new Complex(1, 1), new Complex(2, 3), new Complex(4, 2), new Complex(7, 6), new Complex(11, 5)]);
        const jsResult = diff(a, 2);
        const pyResult = runNumPy(`
result = np.diff(np.array([1+1j, 2+3j, 4+2j, 7+6j, 11+5j]), n=2)
        `);

        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('ediff1d()', () => {
      it('computes ediff1d of complex array matching NumPy', () => {
        const a = array([new Complex(1, 2), new Complex(4, 1), new Complex(6, 5)]);
        const jsResult = ediff1d(a);
        const pyResult = runNumPy(`
result = np.ediff1d(np.array([1+2j, 4+1j, 6+5j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Gradient and Cross Operations
  // ==========================================================================

  describe('Gradient and Cross Operations', () => {
    describe('gradient()', () => {
      it('computes gradient of complex array matching NumPy', () => {
        const f = array([new Complex(1, 1), new Complex(2, 3), new Complex(4, 6), new Complex(7, 10)]);
        const jsResult = gradient(f);
        const pyResult = runNumPy(`
result = np.gradient(np.array([1+1j, 2+3j, 4+6j, 7+10j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose((jsResult as typeof f).toArray(), pyResult.value)).toBe(true);
      });

      it('computes gradient with spacing matching NumPy', () => {
        const f = array([new Complex(0, 0), new Complex(2, 4), new Complex(6, 12)]);
        const jsResult = gradient(f, 2);
        const pyResult = runNumPy(`
result = np.gradient(np.array([0+0j, 2+4j, 6+12j]), 2)
        `);

        expect(arraysClose((jsResult as typeof f).toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('cross()', () => {
      it('computes cross product of complex 3D vectors matching NumPy', () => {
        const a = array([new Complex(1, 1), new Complex(2, 0), new Complex(0, 1)]);
        const b = array([new Complex(0, 1), new Complex(1, 0), new Complex(2, 0)]);
        const jsResult = cross(a, b);
        const pyResult = runNumPy(`
result = np.cross(np.array([1+1j, 2+0j, 0+1j]), np.array([0+1j, 1+0j, 2+0j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('computes cross product of complex 2D vectors matching NumPy', () => {
        const a = array([new Complex(1, 1), new Complex(2, 0)]);
        const b = array([new Complex(0, 1), new Complex(1, 0)]);
        const jsResult = cross(a, b);
        const pyResult = runNumPy(`
result = np.cross(np.array([1+1j, 2+0j]), np.array([0+1j, 1+0j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        // For scalar result, compare directly
        const jsVal = jsResult.toArray();
        const pyVal = pyResult.value;
        expect(jsVal.re).toBeCloseTo(pyVal.re);
        expect(jsVal.im).toBeCloseTo(pyVal.im);
      });
    });
  });

  // ==========================================================================
  // NaN-aware Operations
  // ==========================================================================

  describe('NaN-aware Operations', () => {
    describe('nancumsum()', () => {
      it('computes cumulative sum ignoring NaN matching NumPy', () => {
        const a = array([new Complex(1, 2), new Complex(NaN, NaN), new Complex(3, 4)]);
        const jsResult = nancumsum(a);
        const pyResult = runNumPy(`
result = np.nancumsum(np.array([1+2j, np.nan+np.nan*1j, 3+4j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('nancumprod()', () => {
      it('computes cumulative product ignoring NaN matching NumPy', () => {
        const a = array([new Complex(1, 1), new Complex(NaN, NaN), new Complex(2, 0)]);
        const jsResult = nancumprod(a);
        const pyResult = runNumPy(`
result = np.nancumprod(np.array([1+1j, np.nan+np.nan*1j, 2+0j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('nanvar()', () => {
      it('computes variance ignoring NaN matching NumPy', () => {
        const a = array([new Complex(1, 1), new Complex(NaN, 0), new Complex(2, 2), new Complex(3, 3)]);
        const jsResult = nanvar(a);
        const pyResult = runNumPy(`
result = np.nanvar(np.array([1+1j, np.nan+0j, 2+2j, 3+3j]))
        `);

        expect(typeof jsResult).toBe('number');
        expect(jsResult as number).toBeCloseTo(pyResult.value as number);
      });
    });

    describe('nanstd()', () => {
      it('computes std ignoring NaN matching NumPy', () => {
        const a = array([new Complex(1, 1), new Complex(NaN, 0), new Complex(2, 2), new Complex(3, 3)]);
        const jsResult = nanstd(a);
        const pyResult = runNumPy(`
result = np.nanstd(np.array([1+1j, np.nan+0j, 2+2j, 3+3j]))
        `);

        expect(typeof jsResult).toBe('number');
        expect(jsResult as number).toBeCloseTo(pyResult.value as number);
      });
    });

    describe('nanargmin()', () => {
      it('finds index of min ignoring NaN matching NumPy', () => {
        // NumPy uses lexicographic ordering for complex
        const a = array([new Complex(3, 1), new Complex(NaN, 0), new Complex(1, 2), new Complex(2, 0)]);
        const jsResult = nanargmin(a);
        const pyResult = runNumPy(`
result = np.nanargmin(np.array([3+1j, np.nan+0j, 1+2j, 2+0j]))
        `);

        expect(jsResult).toBe(pyResult.value);
      });
    });

    describe('nanargmax()', () => {
      it('finds index of max ignoring NaN matching NumPy', () => {
        const a = array([new Complex(3, 1), new Complex(NaN, 0), new Complex(1, 2), new Complex(2, 0)]);
        const jsResult = nanargmax(a);
        const pyResult = runNumPy(`
result = np.nanargmax(np.array([3+1j, np.nan+0j, 1+2j, 2+0j]))
        `);

        expect(jsResult).toBe(pyResult.value);
      });
    });
  });

  // ==========================================================================
  // Average and Weighted Average
  // ==========================================================================

  describe('average()', () => {
    it('computes simple average of complex array matching NumPy', () => {
      const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
      const jsResult = average(a);
      const pyResult = runNumPy(`
result = np.average(np.array([1+2j, 3+4j, 5+6j]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });

    it('computes weighted average of complex array matching NumPy', () => {
      const a = array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3)]);
      const jsResult = average(a, undefined, array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.average(np.array([1+1j, 2+2j, 3+3j]), weights=np.array([1, 2, 3]))
      `);

      expect(jsResult).toBeInstanceOf(Complex);
      expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
      expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
    });
  });

  // ==========================================================================
  // Sorting Operations
  // ==========================================================================

  describe('Sorting Operations', () => {
    describe('sort()', () => {
      it('sorts complex array using lexicographic ordering matching NumPy', () => {
        const a = array([new Complex(3, 1), new Complex(1, 2), new Complex(2, 0), new Complex(1, 1)]);
        const jsResult = sort(a);
        const pyResult = runNumPy(`
result = np.sort(np.array([3+1j, 1+2j, 2+0j, 1+1j]))
        `);

        expect(jsResult.dtype).toBe('complex128');
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('argsort()', () => {
      it('returns indices that would sort complex array matching NumPy', () => {
        const a = array([new Complex(3, 1), new Complex(1, 2), new Complex(2, 0), new Complex(1, 1)]);
        const jsResult = argsort(a);
        const pyResult = runNumPy(`
result = np.argsort(np.array([3+1j, 1+2j, 2+0j, 1+1j]))
        `);

        expect(jsResult.dtype).toBe('int32');
        expect(jsResult.toArray()).toEqual(Array.from(pyResult.value));
      });
    });
  });

  // ==========================================================================
  // Min/Max Operations
  // ==========================================================================

  describe('Min/Max Operations', () => {
    describe('max()', () => {
      it('finds max of complex array using lexicographic ordering matching NumPy', () => {
        const a = array([new Complex(1, 2), new Complex(3, 1), new Complex(2, 5)]);
        const jsResult = max(a);
        const pyResult = runNumPy(`
result = np.max(np.array([1+2j, 3+1j, 2+5j]))
        `);

        expect(jsResult).toBeInstanceOf(Complex);
        expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
        expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
      });
    });

    describe('min()', () => {
      it('finds min of complex array using lexicographic ordering matching NumPy', () => {
        const a = array([new Complex(3, 1), new Complex(1, 5), new Complex(2, 0)]);
        const jsResult = min(a);
        const pyResult = runNumPy(`
result = np.min(np.array([3+1j, 1+5j, 2+0j]))
        `);

        expect(jsResult).toBeInstanceOf(Complex);
        expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
        expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
      });
    });

    describe('nanmin()', () => {
      it('finds min ignoring NaN values matching NumPy', () => {
        const a = array([new Complex(NaN, 0), new Complex(3, 1), new Complex(1, 2)]);
        const jsResult = nanmin(a);
        const pyResult = runNumPy(`
result = np.nanmin(np.array([np.nan+0j, 3+1j, 1+2j]))
        `);

        expect(jsResult).toBeInstanceOf(Complex);
        expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
        expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
      });
    });

    describe('nanmax()', () => {
      it('finds max ignoring NaN values matching NumPy', () => {
        const a = array([new Complex(NaN, 0), new Complex(1, 2), new Complex(3, 1)]);
        const jsResult = nanmax(a);
        const pyResult = runNumPy(`
result = np.nanmax(np.array([np.nan+0j, 1+2j, 3+1j]))
        `);

        expect(jsResult).toBeInstanceOf(Complex);
        expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
        expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
      });
    });

    describe('ptp()', () => {
      it('computes peak-to-peak (max - min) matching NumPy', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(2, 1)]);
        const jsResult = ptp(a);
        const pyResult = runNumPy(`
result = np.ptp(np.array([1+2j, 3+4j, 2+1j]))
        `);

        expect(jsResult).toBeInstanceOf(Complex);
        expect((jsResult as Complex).re).toBeCloseTo(pyResult.value.re);
        expect((jsResult as Complex).im).toBeCloseTo(pyResult.value.im);
      });
    });
  });
});
