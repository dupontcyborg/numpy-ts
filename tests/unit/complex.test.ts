/**
 * Unit tests for complex number support
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  ones,
  Complex,
  real,
  imag,
  conj,
  conjugate,
  angle,
  abs,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
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
  cumulative_sum,
  cumprod,
  cumulative_prod,
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
  isneginf,
  isposinf,
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
} from '../../src';

describe('Complex Number Support', () => {
  describe('Complex class', () => {
    it('creates complex numbers', () => {
      const z = new Complex(1, 2);
      expect(z.re).toBe(1);
      expect(z.im).toBe(2);
    });

    it('computes magnitude', () => {
      const z = new Complex(3, 4);
      expect(z.abs()).toBe(5);
    });

    it('computes angle', () => {
      const z = new Complex(1, 1);
      expect(z.angle()).toBeCloseTo(Math.PI / 4);
    });

    it('computes conjugate', () => {
      const z = new Complex(1, 2);
      const conj = z.conj();
      expect(conj.re).toBe(1);
      expect(conj.im).toBe(-2);
    });

    it('performs arithmetic', () => {
      const a = new Complex(1, 2);
      const b = new Complex(3, 4);

      const sum = a.add(b);
      expect(sum.re).toBe(4);
      expect(sum.im).toBe(6);

      const diff = a.sub(b);
      expect(diff.re).toBe(-2);
      expect(diff.im).toBe(-2);

      // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
      const prod = a.mul(b);
      expect(prod.re).toBe(-5);
      expect(prod.im).toBe(10);
    });

    it('converts to string', () => {
      expect(new Complex(1, 2).toString()).toBe('(1+2j)');
      expect(new Complex(1, -2).toString()).toBe('(1-2j)');
      expect(new Complex(1, 0).toString()).toBe('(1+0j)');
    });
  });

  describe('Array creation with complex dtype', () => {
    it('creates zeros with complex128 dtype', () => {
      const arr = zeros([3], 'complex128');
      expect(arr.dtype).toBe('complex128');
      expect(arr.shape).toEqual([3]);
      expect(arr.size).toBe(3);

      const val = arr.get([0]) as Complex;
      expect(val).toBeInstanceOf(Complex);
      expect(val.re).toBe(0);
      expect(val.im).toBe(0);
    });

    it('creates zeros with complex64 dtype', () => {
      const arr = zeros([2, 2], 'complex64');
      expect(arr.dtype).toBe('complex64');
      expect(arr.shape).toEqual([2, 2]);
    });

    it('creates ones with complex128 dtype', () => {
      const arr = ones([3], 'complex128');
      expect(arr.dtype).toBe('complex128');

      const val = arr.get([0]) as Complex;
      expect(val).toBeInstanceOf(Complex);
      expect(val.re).toBe(1);
      expect(val.im).toBe(0);
    });

    it('creates array from Complex values', () => {
      const arr = array([new Complex(1, 2), new Complex(3, 4)]);
      expect(arr.dtype).toBe('complex128');
      expect(arr.shape).toEqual([2]);

      const val0 = arr.get([0]) as Complex;
      expect(val0.re).toBe(1);
      expect(val0.im).toBe(2);

      const val1 = arr.get([1]) as Complex;
      expect(val1.re).toBe(3);
      expect(val1.im).toBe(4);
    });

    it('creates array from {re, im} objects', () => {
      const arr = array([
        { re: 1, im: 2 },
        { re: 3, im: 4 },
      ]);
      expect(arr.dtype).toBe('complex128');

      const val0 = arr.get([0]) as Complex;
      expect(val0.re).toBe(1);
      expect(val0.im).toBe(2);
    });

    it('creates array from real numbers with complex dtype', () => {
      const arr = array([1, 2, 3], 'complex128');
      expect(arr.dtype).toBe('complex128');

      const val0 = arr.get([0]) as Complex;
      expect(val0.re).toBe(1);
      expect(val0.im).toBe(0);
    });
  });

  describe('toArray with complex', () => {
    it('returns Complex objects for complex arrays', () => {
      const arr = array([new Complex(1, 2), new Complex(3, 4)]);
      const result = arr.toArray();

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(Complex);
      expect(result[0].re).toBe(1);
      expect(result[0].im).toBe(2);
    });

    it('works with 2D complex arrays', () => {
      const arr = array([
        [new Complex(1, 0), new Complex(0, 1)],
        [new Complex(0, -1), new Complex(1, 0)],
      ]);

      expect(arr.shape).toEqual([2, 2]);
      const result = arr.toArray();

      expect(result[0][0]).toBeInstanceOf(Complex);
      expect(result[0][0].re).toBe(1);
      expect(result[0][0].im).toBe(0);
      expect(result[0][1].im).toBe(1);
    });
  });

  describe('Element access with complex', () => {
    it('gets complex elements correctly', () => {
      const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);

      const val = arr.get([1]) as Complex;
      expect(val.re).toBe(3);
      expect(val.im).toBe(4);
    });

    it('sets complex elements correctly', () => {
      const arr = zeros([3], 'complex128');

      arr.set([1], new Complex(7, 8));
      const val = arr.get([1]) as Complex;
      expect(val.re).toBe(7);
      expect(val.im).toBe(8);
    });

    it('sets complex elements from {re, im} objects', () => {
      const arr = zeros([3], 'complex128');

      arr.set([0], { re: 10, im: 20 });
      const val = arr.get([0]) as Complex;
      expect(val.re).toBe(10);
      expect(val.im).toBe(20);
    });
  });

  describe('Type inference', () => {
    it('infers complex128 from Complex values', () => {
      const arr = array([new Complex(1, 2)]);
      expect(arr.dtype).toBe('complex128');
    });

    it('infers complex128 from {re, im} objects', () => {
      const arr = array([{ re: 1, im: 2 }]);
      expect(arr.dtype).toBe('complex128');
    });

    it('does not infer complex from plain numbers', () => {
      const arr = array([1, 2, 3]);
      expect(arr.dtype).toBe('float64');
    });
  });

  describe('Copy operations', () => {
    it('copies complex arrays correctly', () => {
      const original = array([new Complex(1, 2), new Complex(3, 4)]);
      const copy = original.copy();

      expect(copy.dtype).toBe('complex128');
      expect(copy.toArray()[0].re).toBe(1);
      expect(copy.toArray()[0].im).toBe(2);

      // Verify it's a true copy
      original.set([0], new Complex(99, 99));
      expect(copy.get([0]).re).toBe(1);
    });
  });

  describe('Complex array operations', () => {
    describe('real()', () => {
      it('extracts real part from complex array', () => {
        const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
        const realPart = real(arr);

        expect(realPart.dtype).toBe('float64');
        expect(realPart.toArray()).toEqual([1, 3, 5]);
      });

      it('returns copy for real arrays', () => {
        const arr = array([1, 2, 3]);
        const realPart = real(arr);

        expect(realPart.dtype).toBe('float64');
        expect(realPart.toArray()).toEqual([1, 2, 3]);
      });
    });

    describe('imag()', () => {
      it('extracts imaginary part from complex array', () => {
        const arr = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
        const imagPart = imag(arr);

        expect(imagPart.dtype).toBe('float64');
        expect(imagPart.toArray()).toEqual([2, 4, 6]);
      });

      it('returns zeros for real arrays', () => {
        const arr = array([1, 2, 3]);
        const imagPart = imag(arr);

        expect(imagPart.dtype).toBe('float64');
        expect(imagPart.toArray()).toEqual([0, 0, 0]);
      });
    });

    describe('conj() / conjugate()', () => {
      it('computes complex conjugate', () => {
        const arr = array([new Complex(1, 2), new Complex(3, -4)]);
        const conjugated = conj(arr);

        expect(conjugated.dtype).toBe('complex128');
        const result = conjugated.toArray();
        expect(result[0].re).toBe(1);
        expect(result[0].im).toBe(-2);
        expect(result[1].re).toBe(3);
        expect(result[1].im).toBe(4);
      });

      it('conjugate is alias for conj', () => {
        expect(conjugate).toBe(conj);
      });

      it('returns copy for real arrays', () => {
        const arr = array([1, 2, 3]);
        const conjugated = conj(arr);

        expect(conjugated.toArray()).toEqual([1, 2, 3]);
      });
    });

    describe('angle()', () => {
      it('computes phase angle in radians', () => {
        const arr = array([new Complex(1, 0), new Complex(0, 1), new Complex(-1, 0), new Complex(0, -1)]);
        const angles = angle(arr);

        expect(angles.dtype).toBe('float64');
        const result = angles.toArray();
        expect(result[0]).toBeCloseTo(0);
        expect(result[1]).toBeCloseTo(Math.PI / 2);
        expect(result[2]).toBeCloseTo(Math.PI);
        expect(result[3]).toBeCloseTo(-Math.PI / 2);
      });

      it('computes phase angle in degrees', () => {
        const arr = array([new Complex(1, 0), new Complex(0, 1)]);
        const angles = angle(arr, true);

        const result = angles.toArray();
        expect(result[0]).toBeCloseTo(0);
        expect(result[1]).toBeCloseTo(90);
      });
    });

    describe('abs() for complex', () => {
      it('computes magnitude of complex numbers', () => {
        const arr = array([new Complex(3, 4), new Complex(5, 12), new Complex(0, 1)]);
        const magnitudes = abs(arr);

        expect(magnitudes.dtype).toBe('float64');
        const result = magnitudes.toArray();
        expect(result[0]).toBe(5);
        expect(result[1]).toBe(13);
        expect(result[2]).toBe(1);
      });
    });
  });

  describe('Type checking functions', () => {
    describe('iscomplexobj()', () => {
      it('returns true for complex arrays', () => {
        const arr = array([new Complex(1, 2)]);
        expect(iscomplexobj(arr)).toBe(true);
      });

      it('returns false for real arrays', () => {
        const arr = array([1, 2, 3]);
        expect(iscomplexobj(arr)).toBe(false);
      });
    });

    describe('isrealobj()', () => {
      it('returns false for complex arrays', () => {
        const arr = array([new Complex(1, 2)]);
        expect(isrealobj(arr)).toBe(false);
      });

      it('returns true for real arrays', () => {
        const arr = array([1, 2, 3]);
        expect(isrealobj(arr)).toBe(true);
      });
    });

    describe('iscomplex()', () => {
      it('returns true for complex elements with non-zero imaginary part', () => {
        const arr = array([new Complex(1, 2), new Complex(3, 0), new Complex(5, 6)]);
        const result = iscomplex(arr);

        expect(result.toArray()).toEqual([1, 0, 1]); // [true, false, true]
      });

      it('returns all false for real arrays', () => {
        const arr = array([1, 2, 3]);
        const result = iscomplex(arr);

        expect(result.toArray()).toEqual([0, 0, 0]);
      });
    });

    describe('isreal()', () => {
      it('returns true for complex elements with zero imaginary part', () => {
        const arr = array([new Complex(1, 2), new Complex(3, 0), new Complex(5, 6)]);
        const result = isreal(arr);

        expect(result.toArray()).toEqual([0, 1, 0]); // [false, true, false]
      });

      it('returns all true for real arrays', () => {
        const arr = array([1, 2, 3]);
        const result = isreal(arr);

        expect(result.toArray()).toEqual([1, 1, 1]);
      });
    });
  });

  describe('Complex arithmetic operations', () => {
    describe('Addition', () => {
      it('adds two complex arrays', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const b = array([new Complex(5, 6), new Complex(7, 8)]);
        const result = a.add(b);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(6);
        expect(values[0].im).toBe(8);
        expect(values[1].re).toBe(10);
        expect(values[1].im).toBe(12);
      });

      it('adds complex array and real array', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const b = array([5, 6]);
        const result = a.add(b);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(6);
        expect(values[0].im).toBe(2);
        expect(values[1].re).toBe(9);
        expect(values[1].im).toBe(4);
      });

      it('adds complex array and scalar', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = a.add(10);

        const values = result.toArray();
        expect(values[0].re).toBe(11);
        expect(values[0].im).toBe(2);
        expect(values[1].re).toBe(13);
        expect(values[1].im).toBe(4);
      });
    });

    describe('Subtraction', () => {
      it('subtracts two complex arrays', () => {
        const a = array([new Complex(5, 6), new Complex(7, 8)]);
        const b = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = a.subtract(b);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(4);
        expect(values[0].im).toBe(4);
        expect(values[1].re).toBe(4);
        expect(values[1].im).toBe(4);
      });

      it('subtracts scalar from complex array', () => {
        const a = array([new Complex(5, 6), new Complex(7, 8)]);
        const result = a.subtract(3);

        const values = result.toArray();
        expect(values[0].re).toBe(2);
        expect(values[0].im).toBe(6);
        expect(values[1].re).toBe(4);
        expect(values[1].im).toBe(8);
      });
    });

    describe('Multiplication', () => {
      it('multiplies two complex arrays', () => {
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        const a = array([new Complex(1, 2)]);
        const b = array([new Complex(3, 4)]);
        const result = a.multiply(b);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(-5);
        expect(values[0].im).toBe(10);
      });

      it('multiplies complex array by scalar', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = a.multiply(2);

        const values = result.toArray();
        expect(values[0].re).toBe(2);
        expect(values[0].im).toBe(4);
        expect(values[1].re).toBe(6);
        expect(values[1].im).toBe(8);
      });

      it('multiplies complex and real arrays', () => {
        // (1+2i) * 3 = 3 + 6i
        const a = array([new Complex(1, 2)]);
        const b = array([3]);
        const result = a.multiply(b);

        const values = result.toArray();
        expect(values[0].re).toBe(3);
        expect(values[0].im).toBe(6);
      });
    });

    describe('Division', () => {
      it('divides two complex arrays', () => {
        // (3+4i)/(1+2i) = (3+4i)(1-2i) / (1+4) = (3 - 6i + 4i + 8) / 5 = (11 - 2i) / 5 = 2.2 - 0.4i
        const a = array([new Complex(3, 4)]);
        const b = array([new Complex(1, 2)]);
        const result = a.divide(b);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2.2);
        expect(values[0].im).toBeCloseTo(-0.4);
      });

      it('divides complex array by scalar', () => {
        const a = array([new Complex(4, 6), new Complex(8, 10)]);
        const result = a.divide(2);

        const values = result.toArray();
        expect(values[0].re).toBe(2);
        expect(values[0].im).toBe(3);
        expect(values[1].re).toBe(4);
        expect(values[1].im).toBe(5);
      });
    });

    describe('Negation', () => {
      it('negates complex array', () => {
        const a = array([new Complex(1, 2), new Complex(-3, 4)]);
        const result = a.negative();

        const values = result.toArray();
        expect(values[0].re).toBe(-1);
        expect(values[0].im).toBe(-2);
        expect(values[1].re).toBe(3);
        expect(values[1].im).toBe(-4);
      });
    });

    describe('sqrt()', () => {
      it('computes sqrt of complex numbers', () => {
        // sqrt(0+i) = sqrt(2)/2 + sqrt(2)/2 * i
        const a = array([new Complex(0, 1)]);
        const result = sqrt(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.SQRT2 / 2);
        expect(values[0].im).toBeCloseTo(Math.SQRT2 / 2);
      });

      it('computes sqrt of negative number as complex', () => {
        // sqrt(-1+0i) = 0 + i
        const a = array([new Complex(-1, 0)]);
        const result = sqrt(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(1);
      });

      it('computes sqrt of positive real as complex', () => {
        // sqrt(4+0i) = 2 + 0i
        const a = array([new Complex(4, 0)]);
        const result = sqrt(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[0].im).toBeCloseTo(0);
      });
    });

    describe('power()', () => {
      it('raises complex to real power', () => {
        // (1+i)^2 = 1 + 2i - 1 = 0 + 2i
        const a = array([new Complex(1, 1)]);
        const result = power(a, 2);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(2);
      });

      it('raises complex to fractional power', () => {
        // (4+0i)^0.5 = 2 + 0i
        const a = array([new Complex(4, 0)]);
        const result = power(a, 0.5);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('raises complex to negative power', () => {
        // (1+0i)^(-1) = 1 + 0i
        const a = array([new Complex(1, 0)]);
        const result = power(a, -1);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });
    });
  });

  describe('Complex comparison operations', () => {
    describe('equal()', () => {
      it('compares complex arrays for equality', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const b = array([new Complex(1, 2), new Complex(3, 5)]);

        // Using NDArray comparison method
        const result = a.equal(b);
        expect(result.toArray()).toEqual([1, 0]); // [true, false]
      });

      it('compares complex with scalar', () => {
        const a = array([new Complex(3, 0), new Complex(3, 1)]);
        const result = a.equal(3);

        // Only 3+0i equals 3
        expect(result.toArray()).toEqual([1, 0]);
      });
    });

    describe('not_equal()', () => {
      it('compares complex arrays for inequality', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const b = array([new Complex(1, 2), new Complex(3, 5)]);

        const result = a.not_equal(b);
        expect(result.toArray()).toEqual([0, 1]); // [false, true]
      });
    });

    describe('greater()', () => {
      it('compares complex arrays with lexicographic ordering', () => {
        // Same real, different imag
        const a = array([new Complex(1, 3)]);
        const b = array([new Complex(1, 2)]);
        expect(a.greater(b).toArray()).toEqual([1]); // 1+3i > 1+2i

        // Different real
        const c = array([new Complex(2, 1)]);
        const d = array([new Complex(1, 10)]);
        expect(c.greater(d).toArray()).toEqual([1]); // 2+1i > 1+10i (real decides)
      });

      it('compares complex with scalar', () => {
        const a = array([new Complex(3, 1), new Complex(2, 0)]);
        // 3+1i > 3? Yes (real equal, imag > 0)
        // 2+0i > 3? No (real 2 < 3)
        expect(a.greater(3).toArray()).toEqual([1, 0]);
      });
    });

    describe('less()', () => {
      it('compares complex arrays with lexicographic ordering', () => {
        const a = array([new Complex(1, 2)]);
        const b = array([new Complex(1, 3)]);
        expect(a.less(b).toArray()).toEqual([1]); // 1+2i < 1+3i

        const c = array([new Complex(1, 10)]);
        const d = array([new Complex(2, 1)]);
        expect(c.less(d).toArray()).toEqual([1]); // 1+10i < 2+1i (real decides)
      });
    });

    describe('greater_equal() and less_equal()', () => {
      it('handles equal complex values', () => {
        const a = array([new Complex(1, 2), new Complex(1, 2)]);
        const b = array([new Complex(1, 2), new Complex(1, 3)]);

        expect(a.greater_equal(b).toArray()).toEqual([1, 0]); // equal, less
        expect(a.less_equal(b).toArray()).toEqual([1, 1]);    // equal, less
      });
    });
  });

  describe('Complex reduction operations', () => {
    describe('sum()', () => {
      it('sums all elements returning Complex', () => {
        // (1+2i) + (3+4i) + (5+6i) = 9 + 12i
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
        const result = a.sum();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBe(9);
        expect((result as Complex).im).toBe(12);
      });

      it('sums along axis', () => {
        // [[1+2i, 3+4i], [5+6i, 7+8i]]
        // sum axis=0: [(1+5)+(2+6)i, (3+7)+(4+8)i] = [6+8i, 10+12i]
        const a = array([
          [new Complex(1, 2), new Complex(3, 4)],
          [new Complex(5, 6), new Complex(7, 8)],
        ]);
        const result = a.sum(0);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(6);
        expect(values[0].im).toBe(8);
        expect(values[1].re).toBe(10);
        expect(values[1].im).toBe(12);
      });
    });

    describe('mean()', () => {
      it('computes mean of all elements returning Complex', () => {
        // (1+2i + 3+4i + 5+6i) / 3 = (9+12i) / 3 = 3 + 4i
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
        const result = a.mean();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBe(3);
        expect((result as Complex).im).toBe(4);
      });

      it('computes mean along axis', () => {
        const a = array([
          [new Complex(2, 4), new Complex(4, 8)],
          [new Complex(6, 12), new Complex(8, 16)],
        ]);
        // mean axis=1: [(2+4)/2 + (4+8)/2 i, ...] = [3+6i, 7+14i]
        const result = a.mean(1);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(3);
        expect(values[0].im).toBe(6);
        expect(values[1].re).toBe(7);
        expect(values[1].im).toBe(14);
      });
    });

    describe('prod()', () => {
      it('computes product of all elements returning Complex', () => {
        // (1+0i) * (2+0i) * (3+0i) = 6 + 0i
        const a = array([new Complex(1, 0), new Complex(2, 0), new Complex(3, 0)]);
        const result = a.prod();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBe(6);
        expect((result as Complex).im).toBe(0);
      });

      it('computes product with imaginary parts', () => {
        // (1+i) * (1+i) = 1 + 2i - 1 = 0 + 2i
        const a = array([new Complex(1, 1), new Complex(1, 1)]);
        const result = a.prod();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBeCloseTo(0);
        expect((result as Complex).im).toBeCloseTo(2);
      });

      it('computes product along axis', () => {
        const a = array([
          [new Complex(1, 0), new Complex(2, 0)],
          [new Complex(3, 0), new Complex(4, 0)],
        ]);
        // prod axis=0: [1*3, 2*4] = [3+0i, 8+0i]
        const result = a.prod(0);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(3);
        expect(values[0].im).toBe(0);
        expect(values[1].re).toBe(8);
        expect(values[1].im).toBe(0);
      });
    });
  });

  describe('Complex exponential and logarithmic operations', () => {
    describe('exp()', () => {
      it('computes exp of purely imaginary number', () => {
        // exp(i*pi) = -1 + 0i (Euler's identity)
        const a = array([new Complex(0, Math.PI)]);
        const result = exp(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(-1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes exp of purely real number', () => {
        // exp(1 + 0i) = e + 0i
        const a = array([new Complex(1, 0)]);
        const result = exp(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.E);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes exp of complex number', () => {
        // exp(1 + i*pi/2) = e * (cos(pi/2) + i*sin(pi/2)) = e * (0 + i) = 0 + e*i
        const a = array([new Complex(1, Math.PI / 2)]);
        const result = exp(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.E);
      });

      it('computes exp of zero', () => {
        // exp(0) = 1
        const a = array([new Complex(0, 0)]);
        const result = exp(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });
    });

    describe('log()', () => {
      it('computes log of positive real number', () => {
        // log(e + 0i) = 1 + 0i
        const a = array([new Complex(Math.E, 0)]);
        const result = log(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log of negative real number', () => {
        // log(-1 + 0i) = 0 + i*pi (principal branch)
        const a = array([new Complex(-1, 0)]);
        const result = log(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.PI);
      });

      it('computes log of purely imaginary number', () => {
        // log(i) = log(|1|) + i*pi/2 = 0 + i*pi/2
        const a = array([new Complex(0, 1)]);
        const result = log(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.PI / 2);
      });

      it('computes log of complex number', () => {
        // log(1+i) = log(sqrt(2)) + i*pi/4
        const a = array([new Complex(1, 1)]);
        const result = log(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.log(Math.SQRT2));
        expect(values[0].im).toBeCloseTo(Math.PI / 4);
      });
    });

    describe('exp2()', () => {
      it('computes 2^z for complex z', () => {
        // 2^(0+0i) = 1
        const a = array([new Complex(0, 0)]);
        const result = exp2(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes 2^1 = 2', () => {
        const a = array([new Complex(1, 0)]);
        const result = exp2(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes 2^i', () => {
        // 2^i = exp(i*ln(2)) = cos(ln2) + i*sin(ln2)
        const a = array([new Complex(0, 1)]);
        const result = exp2(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.cos(Math.LN2));
        expect(values[0].im).toBeCloseTo(Math.sin(Math.LN2));
      });
    });

    describe('expm1()', () => {
      it('computes exp(z) - 1', () => {
        // expm1(0) = 0
        const a = array([new Complex(0, 0)]);
        const result = expm1(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes expm1(i*pi) = -2', () => {
        // exp(i*pi) - 1 = -1 - 1 = -2
        const a = array([new Complex(0, Math.PI)]);
        const result = expm1(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(-2);
        expect(values[0].im).toBeCloseTo(0);
      });
    });

    describe('log2()', () => {
      it('computes log base 2 of complex number', () => {
        // log2(2 + 0i) = 1 + 0i
        const a = array([new Complex(2, 0)]);
        const result = log2(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log2(4) = 2', () => {
        const a = array([new Complex(4, 0)]);
        const result = log2(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log2(-1)', () => {
        // log2(-1) = log(-1) / ln(2) = (0 + i*pi) / ln(2)
        const a = array([new Complex(-1, 0)]);
        const result = log2(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.PI / Math.LN2);
      });
    });

    describe('log10()', () => {
      it('computes log base 10 of complex number', () => {
        // log10(10 + 0i) = 1 + 0i
        const a = array([new Complex(10, 0)]);
        const result = log10(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log10(100) = 2', () => {
        const a = array([new Complex(100, 0)]);
        const result = log10(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log10(-1)', () => {
        // log10(-1) = log(-1) / ln(10) = (0 + i*pi) / ln(10)
        const a = array([new Complex(-1, 0)]);
        const result = log10(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.PI / Math.LN10);
      });
    });

    describe('log1p()', () => {
      it('computes log(1 + z)', () => {
        // log1p(0) = log(1) = 0
        const a = array([new Complex(0, 0)]);
        const result = log1p(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log1p(e-1) = 1', () => {
        // log(1 + (e-1)) = log(e) = 1
        const a = array([new Complex(Math.E - 1, 0)]);
        const result = log1p(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes log1p(i)', () => {
        // log1p(i) = log(1 + i) = log(sqrt(2)) + i*pi/4
        const a = array([new Complex(0, 1)]);
        const result = log1p(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.log(Math.SQRT2));
        expect(values[0].im).toBeCloseTo(Math.PI / 4);
      });

      it('computes log1p(-2) = log(-1)', () => {
        // log(1 + (-2)) = log(-1) = 0 + i*pi
        const a = array([new Complex(-2, 0)]);
        const result = log1p(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.PI);
      });
    });

    describe('exp and log are inverse operations', () => {
      it('exp(log(z)) = z', () => {
        const z = array([new Complex(3, 4)]);
        const result = exp(log(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(3);
        expect(values[0].im).toBeCloseTo(4);
      });

      it('log(exp(z)) = z for small imaginary part', () => {
        // For principal branch, this works when -pi < im(z) <= pi
        const z = array([new Complex(1, 0.5)]);
        const result = log(exp(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0.5);
      });
    });
  });

  // ==========================================================================
  // Complex Trigonometric Operations
  // ==========================================================================
  describe('Complex trigonometric operations', () => {
    describe('sin()', () => {
      it('computes sin(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = sin(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes sin(pi/2) = 1', () => {
        const a = array([new Complex(Math.PI / 2, 0)]);
        const result = sin(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes sin(i) = i*sinh(1)', () => {
        // sin(0 + i) = sin(0)cosh(1) + i*cos(0)sinh(1) = 0 + i*sinh(1)
        const a = array([new Complex(0, 1)]);
        const result = sin(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.sinh(1));
      });

      it('computes sin(a+bi) using correct formula', () => {
        // sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        const a = 1;
        const b = 2;
        const z = array([new Complex(a, b)]);
        const result = sin(z);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.sin(a) * Math.cosh(b));
        expect(values[0].im).toBeCloseTo(Math.cos(a) * Math.sinh(b));
      });
    });

    describe('cos()', () => {
      it('computes cos(0) = 1', () => {
        const a = array([new Complex(0, 0)]);
        const result = cos(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes cos(pi) = -1', () => {
        const a = array([new Complex(Math.PI, 0)]);
        const result = cos(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(-1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes cos(i) = cosh(1)', () => {
        // cos(0 + i) = cos(0)cosh(1) - i*sin(0)sinh(1) = cosh(1) - 0
        const a = array([new Complex(0, 1)]);
        const result = cos(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.cosh(1));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes cos(a+bi) using correct formula', () => {
        // cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        const a = 1;
        const b = 2;
        const z = array([new Complex(a, b)]);
        const result = cos(z);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.cos(a) * Math.cosh(b));
        expect(values[0].im).toBeCloseTo(-Math.sin(a) * Math.sinh(b));
      });
    });

    describe('tan()', () => {
      it('computes tan(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = tan(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes tan(pi/4) = 1', () => {
        const a = array([new Complex(Math.PI / 4, 0)]);
        const result = tan(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes tan(i) = i*tanh(1)', () => {
        // tan(z) = (sin(2a) + i*sinh(2b)) / (cos(2a) + cosh(2b))
        // For z = i: a=0, b=1
        // = (sin(0) + i*sinh(2)) / (cos(0) + cosh(2))
        // = i*sinh(2) / (1 + cosh(2))
        // = i * tanh(1)
        const a = array([new Complex(0, 1)]);
        const result = tan(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.tanh(1));
      });

      it('computes tan(a+bi) using correct formula', () => {
        // tan(z) = (sin(2a) + i*sinh(2b)) / (cos(2a) + cosh(2b))
        const a = 0.5;
        const b = 0.3;
        const z = array([new Complex(a, b)]);
        const result = tan(z);

        const denom = Math.cos(2 * a) + Math.cosh(2 * b);
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.sin(2 * a) / denom);
        expect(values[0].im).toBeCloseTo(Math.sinh(2 * b) / denom);
      });
    });

    describe('arcsin()', () => {
      it('computes arcsin(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = arcsin(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes arcsin(1) = pi/2', () => {
        const a = array([new Complex(1, 0)]);
        const result = arcsin(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.PI / 2);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('arcsin is inverse of sin for small values', () => {
        const z = array([new Complex(0.3, 0.4)]);
        const result = arcsin(sin(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.3);
        expect(values[0].im).toBeCloseTo(0.4);
      });

      it('computes arcsin(2) with imaginary result', () => {
        // arcsin(2) is complex because |2| > 1
        const a = array([new Complex(2, 0)]);
        const result = arcsin(a);

        // sin(arcsin(2)) should equal 2
        const check = sin(result);
        const checkValues = check.toArray();
        expect(checkValues[0].re).toBeCloseTo(2);
        expect(checkValues[0].im).toBeCloseTo(0);
      });
    });

    describe('arccos()', () => {
      it('computes arccos(1) = 0', () => {
        const a = array([new Complex(1, 0)]);
        const result = arccos(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes arccos(0) = pi/2', () => {
        const a = array([new Complex(0, 0)]);
        const result = arccos(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.PI / 2);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('arccos is inverse of cos for values in [0, pi]', () => {
        const z = array([new Complex(0.5, 0.3)]);
        const result = arccos(cos(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.5);
        expect(values[0].im).toBeCloseTo(0.3);
      });

      it('computes arccos(2) with imaginary result', () => {
        // arccos(2) is complex because |2| > 1
        const a = array([new Complex(2, 0)]);
        const result = arccos(a);

        // cos(arccos(2)) should equal 2
        const check = cos(result);
        const checkValues = check.toArray();
        expect(checkValues[0].re).toBeCloseTo(2);
        expect(checkValues[0].im).toBeCloseTo(0);
      });
    });

    describe('arctan()', () => {
      it('computes arctan(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = arctan(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes arctan(1) = pi/4', () => {
        const a = array([new Complex(1, 0)]);
        const result = arctan(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.PI / 4);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('arctan is inverse of tan for small values', () => {
        const z = array([new Complex(0.3, 0.2)]);
        const result = arctan(tan(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.3);
        expect(values[0].im).toBeCloseTo(0.2);
      });

      it('computes arctan(2i) correctly', () => {
        // arctan(z) is special at z = i and z = -i (poles)
        // For z = 2i, result should be well-defined
        const a = array([new Complex(0, 2)]);
        const result = arctan(a);

        // tan(arctan(2i)) should equal 2i
        const check = tan(result);
        const checkValues = check.toArray();
        expect(checkValues[0].re).toBeCloseTo(0);
        expect(checkValues[0].im).toBeCloseTo(2);
      });
    });

    describe('sin and arcsin are inverse operations', () => {
      it('arcsin(sin(z)) = z for small z', () => {
        const z = array([new Complex(0.5, 0.5)]);
        const result = arcsin(sin(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.5);
        expect(values[0].im).toBeCloseTo(0.5);
      });

      it('sin(arcsin(z)) = z', () => {
        const z = array([new Complex(0.7, 0.3)]);
        const result = sin(arcsin(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.7);
        expect(values[0].im).toBeCloseTo(0.3);
      });
    });

    describe('cos and arccos are inverse operations', () => {
      it('arccos(cos(z)) = z for z in principal range', () => {
        const z = array([new Complex(1.0, 0.5)]);
        const result = arccos(cos(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1.0);
        expect(values[0].im).toBeCloseTo(0.5);
      });

      it('cos(arccos(z)) = z', () => {
        const z = array([new Complex(0.5, 0.5)]);
        const result = cos(arccos(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.5);
        expect(values[0].im).toBeCloseTo(0.5);
      });
    });

    describe('tan and arctan are inverse operations', () => {
      it('arctan(tan(z)) = z for small z', () => {
        const z = array([new Complex(0.5, 0.3)]);
        const result = arctan(tan(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.5);
        expect(values[0].im).toBeCloseTo(0.3);
      });

      it('tan(arctan(z)) = z', () => {
        const z = array([new Complex(1.5, 0.5)]);
        const result = tan(arctan(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1.5);
        expect(values[0].im).toBeCloseTo(0.5);
      });
    });

    describe('Pythagorean identity', () => {
      it('sin²(z) + cos²(z) = 1 for complex z', () => {
        const z = array([new Complex(1.2, 0.8)]);
        const sinZ = sin(z);
        const cosZ = cos(z);

        // Compute sin²(z) + cos²(z)
        const sinVals = sinZ.toArray();
        const cosVals = cosZ.toArray();

        // sin²(z) = (a + bi)² = a² - b² + 2abi
        const sin2Re =
          sinVals[0].re * sinVals[0].re - sinVals[0].im * sinVals[0].im;
        const sin2Im = 2 * sinVals[0].re * sinVals[0].im;

        // cos²(z) = (c + di)² = c² - d² + 2cdi
        const cos2Re =
          cosVals[0].re * cosVals[0].re - cosVals[0].im * cosVals[0].im;
        const cos2Im = 2 * cosVals[0].re * cosVals[0].im;

        // sin² + cos² should equal 1
        expect(sin2Re + cos2Re).toBeCloseTo(1);
        expect(sin2Im + cos2Im).toBeCloseTo(0);
      });
    });
  });

  // ==========================================================================
  // Complex Hyperbolic Operations
  // ==========================================================================
  describe('Complex hyperbolic operations', () => {
    describe('sinh()', () => {
      it('computes sinh(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = sinh(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes sinh(1) for real input', () => {
        const a = array([new Complex(1, 0)]);
        const result = sinh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.sinh(1));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes sinh(i) = i*sin(1)', () => {
        // sinh(0 + i) = sinh(0)cos(1) + i*cosh(0)sin(1) = 0 + i*sin(1)
        const a = array([new Complex(0, 1)]);
        const result = sinh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.sin(1));
      });

      it('computes sinh(a+bi) using correct formula', () => {
        // sinh(a+bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
        const a = 1;
        const b = 2;
        const z = array([new Complex(a, b)]);
        const result = sinh(z);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.sinh(a) * Math.cos(b));
        expect(values[0].im).toBeCloseTo(Math.cosh(a) * Math.sin(b));
      });
    });

    describe('cosh()', () => {
      it('computes cosh(0) = 1', () => {
        const a = array([new Complex(0, 0)]);
        const result = cosh(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes cosh(1) for real input', () => {
        const a = array([new Complex(1, 0)]);
        const result = cosh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.cosh(1));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes cosh(i) = cos(1)', () => {
        // cosh(0 + i) = cosh(0)cos(1) + i*sinh(0)sin(1) = cos(1) + 0
        const a = array([new Complex(0, 1)]);
        const result = cosh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.cos(1));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes cosh(a+bi) using correct formula', () => {
        // cosh(a+bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
        const a = 1;
        const b = 2;
        const z = array([new Complex(a, b)]);
        const result = cosh(z);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.cosh(a) * Math.cos(b));
        expect(values[0].im).toBeCloseTo(Math.sinh(a) * Math.sin(b));
      });
    });

    describe('tanh()', () => {
      it('computes tanh(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = tanh(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes tanh(1) for real input', () => {
        const a = array([new Complex(1, 0)]);
        const result = tanh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.tanh(1));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes tanh(i) = i*tan(1)', () => {
        // tanh(z) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))
        // For z = i: a=0, b=1
        // = (sinh(0) + i*sin(2)) / (cosh(0) + cos(2))
        // = i*sin(2) / (1 + cos(2))
        // = i * tan(1)
        const a = array([new Complex(0, 1)]);
        const result = tanh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(Math.tan(1));
      });

      it('computes tanh(a+bi) using correct formula', () => {
        // tanh(z) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))
        const a = 0.5;
        const b = 0.3;
        const z = array([new Complex(a, b)]);
        const result = tanh(z);

        const denom = Math.cosh(2 * a) + Math.cos(2 * b);
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.sinh(2 * a) / denom);
        expect(values[0].im).toBeCloseTo(Math.sin(2 * b) / denom);
      });
    });

    describe('arcsinh()', () => {
      it('computes arcsinh(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = arcsinh(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes arcsinh(1) for real input', () => {
        const a = array([new Complex(1, 0)]);
        const result = arcsinh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.asinh(1));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('arcsinh is inverse of sinh for small values', () => {
        const z = array([new Complex(0.3, 0.4)]);
        const result = arcsinh(sinh(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.3);
        expect(values[0].im).toBeCloseTo(0.4);
      });

      it('sinh(arcsinh(z)) = z', () => {
        const z = array([new Complex(1.5, 0.5)]);
        const result = sinh(arcsinh(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1.5);
        expect(values[0].im).toBeCloseTo(0.5);
      });
    });

    describe('arccosh()', () => {
      it('computes arccosh(1) = 0', () => {
        const a = array([new Complex(1, 0)]);
        const result = arccosh(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes arccosh(2) for real input', () => {
        const a = array([new Complex(2, 0)]);
        const result = arccosh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.acosh(2));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('cosh(arccosh(z)) = z', () => {
        const z = array([new Complex(2, 0.5)]);
        const result = cosh(arccosh(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[0].im).toBeCloseTo(0.5);
      });

      it('computes arccosh of complex value', () => {
        const z = array([new Complex(0.5, 0.3)]);
        const result = arccosh(z);

        // Verify by checking cosh(result) = z
        const check = cosh(result);
        const checkValues = check.toArray();
        expect(checkValues[0].re).toBeCloseTo(0.5);
        expect(checkValues[0].im).toBeCloseTo(0.3);
      });
    });

    describe('arctanh()', () => {
      it('computes arctanh(0) = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = arctanh(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes arctanh(0.5) for real input', () => {
        const a = array([new Complex(0.5, 0)]);
        const result = arctanh(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(Math.atanh(0.5));
        expect(values[0].im).toBeCloseTo(0);
      });

      it('arctanh is inverse of tanh for small values', () => {
        const z = array([new Complex(0.3, 0.2)]);
        const result = arctanh(tanh(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.3);
        expect(values[0].im).toBeCloseTo(0.2);
      });

      it('tanh(arctanh(z)) = z', () => {
        const z = array([new Complex(0.5, 0.5)]);
        const result = tanh(arctanh(z));

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0.5);
        expect(values[0].im).toBeCloseTo(0.5);
      });
    });

    describe('Hyperbolic identity', () => {
      it('cosh²(z) - sinh²(z) = 1 for complex z', () => {
        const z = array([new Complex(1.2, 0.8)]);
        const sinhZ = sinh(z);
        const coshZ = cosh(z);

        // Compute cosh²(z) - sinh²(z)
        const sinhVals = sinhZ.toArray();
        const coshVals = coshZ.toArray();

        // sinh²(z) = (a + bi)² = a² - b² + 2abi
        const sinh2Re =
          sinhVals[0].re * sinhVals[0].re - sinhVals[0].im * sinhVals[0].im;
        const sinh2Im = 2 * sinhVals[0].re * sinhVals[0].im;

        // cosh²(z) = (c + di)² = c² - d² + 2cdi
        const cosh2Re =
          coshVals[0].re * coshVals[0].re - coshVals[0].im * coshVals[0].im;
        const cosh2Im = 2 * coshVals[0].re * coshVals[0].im;

        // cosh² - sinh² should equal 1
        expect(cosh2Re - sinh2Re).toBeCloseTo(1);
        expect(cosh2Im - sinh2Im).toBeCloseTo(0);
      });
    });

    describe('Relationship between trig and hyperbolic', () => {
      it('sinh(iz) = i*sin(z)', () => {
        const z = array([new Complex(0.5, 0.3)]);
        // iz = -0.3 + 0.5i
        const iz = array([new Complex(-0.3, 0.5)]);

        const sinhIz = sinh(iz);
        const sinZ = sin(z);

        const sinhVals = sinhIz.toArray();
        const sinVals = sinZ.toArray();

        // i * sin(z) = i * (a + bi) = -b + ai
        expect(sinhVals[0].re).toBeCloseTo(-sinVals[0].im);
        expect(sinhVals[0].im).toBeCloseTo(sinVals[0].re);
      });

      it('cosh(iz) = cos(z)', () => {
        const z = array([new Complex(0.5, 0.3)]);
        // iz = -0.3 + 0.5i
        const iz = array([new Complex(-0.3, 0.5)]);

        const coshIz = cosh(iz);
        const cosZ = cos(z);

        const coshVals = coshIz.toArray();
        const cosVals = cosZ.toArray();

        expect(coshVals[0].re).toBeCloseTo(cosVals[0].re);
        expect(coshVals[0].im).toBeCloseTo(cosVals[0].im);
      });
    });
  });

  // ==========================================================================
  // Complex Arithmetic Operations
  // ==========================================================================
  describe('Complex arithmetic operations', () => {
    describe('positive()', () => {
      it('returns a copy of the complex array', () => {
        const a = array([new Complex(3, 4), new Complex(-1, 2)]);
        const result = positive(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBe(3);
        expect(values[0].im).toBe(4);
        expect(values[1].re).toBe(-1);
        expect(values[1].im).toBe(2);
      });

      it('preserves zeros', () => {
        const a = array([new Complex(0, 0)]);
        const result = positive(a);

        const values = result.toArray();
        expect(values[0].re).toBe(0);
        expect(values[0].im).toBe(0);
      });

      it('preserves negative values', () => {
        const a = array([new Complex(-5, -3)]);
        const result = positive(a);

        const values = result.toArray();
        expect(values[0].re).toBe(-5);
        expect(values[0].im).toBe(-3);
      });
    });

    describe('reciprocal()', () => {
      it('computes 1/z for complex numbers', () => {
        // 1/(3+4i) = (3-4i)/(9+16) = (3-4i)/25
        const a = array([new Complex(3, 4)]);
        const result = reciprocal(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(3 / 25);
        expect(values[0].im).toBeCloseTo(-4 / 25);
      });

      it('computes 1/1 = 1', () => {
        const a = array([new Complex(1, 0)]);
        const result = reciprocal(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes 1/i = -i', () => {
        const a = array([new Complex(0, 1)]);
        const result = reciprocal(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(-1);
      });

      it('z * reciprocal(z) = 1', () => {
        const z = array([new Complex(2, 3)]);
        const recip = reciprocal(z);

        // Manually compute z * recip
        const zVals = z.toArray();
        const recipVals = recip.toArray();

        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        const prodRe =
          zVals[0].re * recipVals[0].re - zVals[0].im * recipVals[0].im;
        const prodIm =
          zVals[0].re * recipVals[0].im + zVals[0].im * recipVals[0].re;

        expect(prodRe).toBeCloseTo(1);
        expect(prodIm).toBeCloseTo(0);
      });

      it('computes reciprocal of purely imaginary number', () => {
        // 1/(2i) = -i/2
        const a = array([new Complex(0, 2)]);
        const result = reciprocal(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(-0.5);
      });
    });

    describe('square()', () => {
      it('computes z² for complex numbers', () => {
        // (3+4i)² = 9 + 24i - 16 = -7 + 24i
        const a = array([new Complex(3, 4)]);
        const result = square(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(-7);
        expect(values[0].im).toBeCloseTo(24);
      });

      it('computes 1² = 1', () => {
        const a = array([new Complex(1, 0)]);
        const result = square(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes i² = -1', () => {
        const a = array([new Complex(0, 1)]);
        const result = square(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(-1);
        expect(values[0].im).toBeCloseTo(0);
      });

      it('computes 0² = 0', () => {
        const a = array([new Complex(0, 0)]);
        const result = square(a);

        const values = result.toArray();
        expect(values[0].re).toBe(0);
        expect(values[0].im).toBe(0);
      });

      it('computes (1+i)² = 2i', () => {
        // (1+i)² = 1 + 2i - 1 = 2i
        const a = array([new Complex(1, 1)]);
        const result = square(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(2);
      });

      it('computes square of negative numbers', () => {
        // (-2 - 3i)² = 4 + 12i - 9 = -5 + 12i
        const a = array([new Complex(-2, -3)]);
        const result = square(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(-5);
        expect(values[0].im).toBeCloseTo(12);
      });

      it('sqrt(square(z)) = |z| for real positive z', () => {
        const z = array([new Complex(5, 0)]);
        const sq = square(z);
        const result = sqrt(sq);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(5);
        expect(values[0].im).toBeCloseTo(0);
      });
    });
  });

  // ==========================================================================
  // Complex Reduction Operations
  // ==========================================================================
  describe('Complex reduction operations', () => {
    describe('sum()', () => {
      it('computes sum of complex array', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
        const result = a.sum();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBe(9);
        expect((result as Complex).im).toBe(12);
      });

      it('computes sum of zeros', () => {
        const a = array([new Complex(0, 0), new Complex(0, 0)]);
        const result = a.sum();

        expect((result as Complex).re).toBe(0);
        expect((result as Complex).im).toBe(0);
      });
    });

    describe('prod()', () => {
      it('computes product of complex array', () => {
        // (1+i)(2+i) = 2 + i + 2i - 1 = 1 + 3i
        const a = array([new Complex(1, 1), new Complex(2, 1)]);
        const result = a.prod();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBeCloseTo(1);
        expect((result as Complex).im).toBeCloseTo(3);
      });

      it('product with identity', () => {
        // (3+4i) * 1 = 3+4i
        const a = array([new Complex(3, 4), new Complex(1, 0)]);
        const result = a.prod();

        expect((result as Complex).re).toBeCloseTo(3);
        expect((result as Complex).im).toBeCloseTo(4);
      });
    });

    describe('mean()', () => {
      it('computes mean of complex array', () => {
        const a = array([new Complex(2, 4), new Complex(4, 8), new Complex(6, 12)]);
        const result = a.mean();

        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBeCloseTo(4);
        expect((result as Complex).im).toBeCloseTo(8);
      });

      it('mean of single element', () => {
        const a = array([new Complex(3, 5)]);
        const result = a.mean();

        expect((result as Complex).re).toBeCloseTo(3);
        expect((result as Complex).im).toBeCloseTo(5);
      });
    });

    describe('var()', () => {
      it('computes variance of complex array', () => {
        // Variance of complex: Var(X) = E[|X - μ|²]
        // For [1+1i, 2+2i, 3+3i], mean = 2+2i
        // |1+1i - 2+2i|² = |-1-1i|² = 1 + 1 = 2
        // |2+2i - 2+2i|² = |0|² = 0
        // |3+3i - 2+2i|² = |1+1i|² = 1 + 1 = 2
        // Var = (2 + 0 + 2) / 3 = 4/3
        const a = array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3)]);
        const result = a.var();

        expect(typeof result).toBe('number');
        expect(result as number).toBeCloseTo(4 / 3);
      });

      it('variance of constant array is zero', () => {
        const a = array([new Complex(2, 3), new Complex(2, 3), new Complex(2, 3)]);
        const result = a.var();

        expect(result as number).toBeCloseTo(0);
      });
    });

    describe('std()', () => {
      it('computes std of complex array', () => {
        // std = sqrt(var)
        const a = array([new Complex(1, 1), new Complex(2, 2), new Complex(3, 3)]);
        const result = a.std();

        expect(typeof result).toBe('number');
        expect(result as number).toBeCloseTo(Math.sqrt(4 / 3));
      });

      it('std of constant array is zero', () => {
        const a = array([new Complex(2, 3), new Complex(2, 3)]);
        const result = a.std();

        expect(result as number).toBeCloseTo(0);
      });
    });

    describe('cumsum()', () => {
      it('computes cumulative sum of complex array', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)]);
        const result = cumsum(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        // cumsum: [1+2i, 4+6i, 9+12i]
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(2);
        expect(values[1].re).toBeCloseTo(4);
        expect(values[1].im).toBeCloseTo(6);
        expect(values[2].re).toBeCloseTo(9);
        expect(values[2].im).toBeCloseTo(12);
      });

      it('cumulative_sum is alias for cumsum', () => {
        const a = array([new Complex(1, 1), new Complex(2, 2)]);
        const result1 = cumsum(a).toArray();
        const result2 = cumulative_sum(a).toArray();

        expect(result1[0].re).toBe(result2[0].re);
        expect(result1[1].re).toBe(result2[1].re);
      });
    });

    describe('cumprod()', () => {
      it('computes cumulative product of complex array', () => {
        // (1+i), (1+i)(2+0i) = 2+2i, (2+2i)(1+i) = 2+2i+2i-2 = 0+4i
        const a = array([new Complex(1, 1), new Complex(2, 0), new Complex(1, 1)]);
        const result = cumprod(a);

        expect(result.dtype).toBe('complex128');
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(1);
        expect(values[1].re).toBeCloseTo(2);
        expect(values[1].im).toBeCloseTo(2);
        expect(values[2].re).toBeCloseTo(0);
        expect(values[2].im).toBeCloseTo(4);
      });

      it('cumulative_prod is alias for cumprod', () => {
        const a = array([new Complex(1, 0), new Complex(2, 0)]);
        const result1 = cumprod(a).toArray();
        const result2 = cumulative_prod(a).toArray();

        expect(result1[0].re).toBe(result2[0].re);
        expect(result1[1].re).toBe(result2[1].re);
      });

      it('cumprod with identity element', () => {
        const a = array([new Complex(3, 4), new Complex(1, 0)]);
        const result = cumprod(a);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(3);
        expect(values[0].im).toBeCloseTo(4);
        expect(values[1].re).toBeCloseTo(3);
        expect(values[1].im).toBeCloseTo(4);
      });
    });
  });

  describe('Linear Algebra Operations', () => {
    describe('dot()', () => {
      it('computes dot product of two complex 1D arrays', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const b = array([new Complex(5, 6), new Complex(7, 8)]);
        const result = dot(a, b);

        // (1+2i)(5+6i) + (3+4i)(7+8i)
        // = (5-12 + i(6+10)) + (21-32 + i(24+28))
        // = (-7 + 16i) + (-11 + 52i)
        // = -18 + 68i
        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBeCloseTo(-18);
        expect((result as Complex).im).toBeCloseTo(68);
      });

      it('computes 2D x 1D dot product (matrix-vector)', () => {
        const A = array([
          [new Complex(1, 0), new Complex(0, 1)],
          [new Complex(0, -1), new Complex(1, 0)],
        ]);
        const v = array([new Complex(1, 0), new Complex(0, 1)]);
        const result = dot(A, v);

        // First row: (1+0i)(1+0i) + (0+i)(0+i) = 1 + (-1) = 0
        // Second row: (0-i)(1+0i) + (1+0i)(0+i) = -i + i = 0
        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);
        expect(values[1].re).toBeCloseTo(0);
        expect(values[1].im).toBeCloseTo(0);
      });

      it('computes 2D x 2D dot product (matrix multiplication)', () => {
        const A = array([
          [new Complex(1, 1), new Complex(0, 0)],
          [new Complex(0, 0), new Complex(1, -1)],
        ]);
        const B = array([
          [new Complex(2, 0), new Complex(0, 0)],
          [new Complex(0, 0), new Complex(2, 0)],
        ]);
        const result = dot(A, B);

        // Diagonal matrix times scalar matrix
        const values = result.toArray();
        expect(values[0][0].re).toBeCloseTo(2);
        expect(values[0][0].im).toBeCloseTo(2);
        expect(values[1][1].re).toBeCloseTo(2);
        expect(values[1][1].im).toBeCloseTo(-2);
      });
    });

    describe('trace()', () => {
      it('computes trace of complex matrix', () => {
        const A = array([
          [new Complex(1, 2), new Complex(3, 4)],
          [new Complex(5, 6), new Complex(7, 8)],
        ]);
        const result = trace(A);

        // Trace = (1+2i) + (7+8i) = 8 + 10i
        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBeCloseTo(8);
        expect((result as Complex).im).toBeCloseTo(10);
      });

      it('computes trace of identity-like complex matrix', () => {
        const I = array([
          [new Complex(1, 0), new Complex(0, 0)],
          [new Complex(0, 0), new Complex(1, 0)],
        ]);
        const result = trace(I);

        expect((result as Complex).re).toBeCloseTo(2);
        expect((result as Complex).im).toBeCloseTo(0);
      });
    });

    describe('diagonal()', () => {
      it('extracts diagonal from complex matrix', () => {
        const A = array([
          [new Complex(1, 2), new Complex(3, 4)],
          [new Complex(5, 6), new Complex(7, 8)],
        ]);
        const result = diagonal(A);

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(2);
        expect(values[1].re).toBeCloseTo(7);
        expect(values[1].im).toBeCloseTo(8);
      });

      it('extracts off-diagonal from complex matrix', () => {
        const A = array([
          [new Complex(1, 0), new Complex(2, 0), new Complex(3, 0)],
          [new Complex(4, 0), new Complex(5, 0), new Complex(6, 0)],
          [new Complex(7, 0), new Complex(8, 0), new Complex(9, 0)],
        ]);
        const result = diagonal(A, 1); // k=1 is superdiagonal

        const values = result.toArray();
        expect(values[0].re).toBeCloseTo(2);
        expect(values[1].re).toBeCloseTo(6);
      });
    });

    describe('transpose()', () => {
      it('transposes complex matrix', () => {
        const A = array([
          [new Complex(1, 2), new Complex(3, 4)],
          [new Complex(5, 6), new Complex(7, 8)],
        ]);
        const result = transpose(A);

        const values = result.toArray();
        // Transposed: [[1+2i, 5+6i], [3+4i, 7+8i]]
        expect(values[0][0].re).toBeCloseTo(1);
        expect(values[0][0].im).toBeCloseTo(2);
        expect(values[0][1].re).toBeCloseTo(5);
        expect(values[0][1].im).toBeCloseTo(6);
        expect(values[1][0].re).toBeCloseTo(3);
        expect(values[1][0].im).toBeCloseTo(4);
        expect(values[1][1].re).toBeCloseTo(7);
        expect(values[1][1].im).toBeCloseTo(8);
      });
    });

    describe('inner()', () => {
      it('computes inner product of complex 1D arrays', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const b = array([new Complex(5, 6), new Complex(7, 8)]);
        const result = inner(a, b);

        // Same as dot for 1D: -18 + 68i
        expect(result).toBeInstanceOf(Complex);
        expect((result as Complex).re).toBeCloseTo(-18);
        expect((result as Complex).im).toBeCloseTo(68);
      });
    });

    describe('outer()', () => {
      it('computes outer product of complex arrays', () => {
        const a = array([new Complex(1, 0), new Complex(0, 1)]);
        const b = array([new Complex(1, 0), new Complex(0, 1)]);
        const result = outer(a, b);

        // [[1*1, 1*i], [i*1, i*i]] = [[1, i], [i, -1]]
        const values = result.toArray();
        expect(values[0][0].re).toBeCloseTo(1);
        expect(values[0][0].im).toBeCloseTo(0);
        expect(values[0][1].re).toBeCloseTo(0);
        expect(values[0][1].im).toBeCloseTo(1);
        expect(values[1][0].re).toBeCloseTo(0);
        expect(values[1][0].im).toBeCloseTo(1);
        expect(values[1][1].re).toBeCloseTo(-1);
        expect(values[1][1].im).toBeCloseTo(0);
      });
    });

    describe('kron()', () => {
      it('computes Kronecker product of complex matrices', () => {
        const A = array([[new Complex(1, 0), new Complex(0, 1)]]);
        const B = array([[new Complex(1, 0)], [new Complex(0, 1)]]);
        const result = kron(A, B);

        // kron([[1, i]], [[1], [i]]) = [[1, i], [i, -1]]
        const values = result.toArray();
        expect(values[0][0].re).toBeCloseTo(1);
        expect(values[0][0].im).toBeCloseTo(0);
        expect(values[0][1].re).toBeCloseTo(0);
        expect(values[0][1].im).toBeCloseTo(1);
        expect(values[1][0].re).toBeCloseTo(0);
        expect(values[1][0].im).toBeCloseTo(1);
        expect(values[1][1].re).toBeCloseTo(-1);
        expect(values[1][1].im).toBeCloseTo(0);
      });
    });
  });

  describe('Logic/Checking Operations', () => {
    describe('isfinite()', () => {
      it('returns true for finite complex numbers', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = isfinite(a);

        expect(result.toArray()).toEqual([1, 1]);
      });

      it('returns false when real part is infinite', () => {
        const a = array([new Complex(Infinity, 0), new Complex(1, 2)]);
        const result = isfinite(a);

        expect(result.toArray()).toEqual([0, 1]);
      });

      it('returns false when imaginary part is infinite', () => {
        const a = array([new Complex(1, Infinity), new Complex(1, 2)]);
        const result = isfinite(a);

        expect(result.toArray()).toEqual([0, 1]);
      });

      it('returns false for NaN in complex', () => {
        const a = array([new Complex(NaN, 0), new Complex(0, NaN)]);
        const result = isfinite(a);

        expect(result.toArray()).toEqual([0, 0]);
      });
    });

    describe('isinf()', () => {
      it('returns false for finite complex numbers', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = isinf(a);

        expect(result.toArray()).toEqual([0, 0]);
      });

      it('returns true when real part is infinite', () => {
        const a = array([new Complex(Infinity, 0), new Complex(-Infinity, 1)]);
        const result = isinf(a);

        expect(result.toArray()).toEqual([1, 1]);
      });

      it('returns true when imaginary part is infinite', () => {
        const a = array([new Complex(1, Infinity), new Complex(0, -Infinity)]);
        const result = isinf(a);

        expect(result.toArray()).toEqual([1, 1]);
      });

      it('returns false for NaN (NaN is not infinity)', () => {
        const a = array([new Complex(NaN, 0), new Complex(0, NaN)]);
        const result = isinf(a);

        expect(result.toArray()).toEqual([0, 0]);
      });
    });

    describe('isnan()', () => {
      it('returns false for finite complex numbers', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = isnan(a);

        expect(result.toArray()).toEqual([0, 0]);
      });

      it('returns true when real part is NaN', () => {
        const a = array([new Complex(NaN, 0), new Complex(1, 2)]);
        const result = isnan(a);

        expect(result.toArray()).toEqual([1, 0]);
      });

      it('returns true when imaginary part is NaN', () => {
        const a = array([new Complex(0, NaN), new Complex(1, 2)]);
        const result = isnan(a);

        expect(result.toArray()).toEqual([1, 0]);
      });

      it('returns false for infinity (infinity is not NaN)', () => {
        const a = array([new Complex(Infinity, 0), new Complex(0, -Infinity)]);
        const result = isnan(a);

        expect(result.toArray()).toEqual([0, 0]);
      });
    });

    describe('isneginf()', () => {
      it('returns false for finite complex numbers', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = isneginf(a);

        expect(result.toArray()).toEqual([0, 0]);
      });

      it('returns true when real part is -Infinity', () => {
        const a = array([new Complex(-Infinity, 0), new Complex(1, 2)]);
        const result = isneginf(a);

        expect(result.toArray()).toEqual([1, 0]);
      });

      it('returns true when imaginary part is -Infinity', () => {
        const a = array([new Complex(0, -Infinity), new Complex(1, 2)]);
        const result = isneginf(a);

        expect(result.toArray()).toEqual([1, 0]);
      });

      it('returns false for +Infinity', () => {
        const a = array([new Complex(Infinity, 0), new Complex(0, Infinity)]);
        const result = isneginf(a);

        expect(result.toArray()).toEqual([0, 0]);
      });
    });

    describe('isposinf()', () => {
      it('returns false for finite complex numbers', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = isposinf(a);

        expect(result.toArray()).toEqual([0, 0]);
      });

      it('returns true when real part is +Infinity', () => {
        const a = array([new Complex(Infinity, 0), new Complex(1, 2)]);
        const result = isposinf(a);

        expect(result.toArray()).toEqual([1, 0]);
      });

      it('returns true when imaginary part is +Infinity', () => {
        const a = array([new Complex(0, Infinity), new Complex(1, 2)]);
        const result = isposinf(a);

        expect(result.toArray()).toEqual([1, 0]);
      });

      it('returns false for -Infinity', () => {
        const a = array([new Complex(-Infinity, 0), new Complex(0, -Infinity)]);
        const result = isposinf(a);

        expect(result.toArray()).toEqual([0, 0]);
      });
    });
  });

  describe('Searching/Indexing Operations', () => {
    describe('nonzero()', () => {
      it('finds indices of non-zero complex numbers', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const result = nonzero(a);

        expect(result.length).toBe(1); // 1D array returns tuple with 1 element
        expect(result[0].toArray()).toEqual([1, 2]); // indices 1 and 2 are nonzero
      });

      it('treats zero real but nonzero imag as nonzero', () => {
        const a = array([new Complex(0, 0), new Complex(0, 1)]);
        const result = nonzero(a);

        expect(result[0].toArray()).toEqual([1]);
      });

      it('treats nonzero real but zero imag as nonzero', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0)]);
        const result = nonzero(a);

        expect(result[0].toArray()).toEqual([1]);
      });
    });

    describe('argwhere()', () => {
      it('finds indices of non-zero complex numbers', () => {
        const a = array([new Complex(0, 0), new Complex(1, 2), new Complex(0, 0), new Complex(3, 0)]);
        const result = argwhere(a);

        expect(result.shape).toEqual([2, 1]); // 2 nonzero elements, 1D array
        expect(result.toArray()).toEqual([[1], [3]]);
      });
    });

    describe('flatnonzero()', () => {
      it('finds flat indices of non-zero complex numbers', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const result = flatnonzero(a);

        expect(result.toArray()).toEqual([1, 2]);
      });

      it('returns empty array for all zeros', () => {
        const a = array([new Complex(0, 0), new Complex(0, 0)]);
        const result = flatnonzero(a);

        expect(result.toArray()).toEqual([]);
      });
    });

    describe('where()', () => {
      it('returns nonzero indices when only condition given', () => {
        const a = array([new Complex(0, 0), new Complex(1, 2), new Complex(0, 0)]);
        const result = where(a) as ReturnType<typeof nonzero>;

        expect(result[0].toArray()).toEqual([1]);
      });

      it('selects from x or y based on complex condition', () => {
        const cond = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1)]);
        const x = array([new Complex(10, 10), new Complex(20, 20), new Complex(30, 30)]);
        const y = array([new Complex(100, 100), new Complex(200, 200), new Complex(300, 300)]);
        const result = where(cond, x, y);

        const values = (result as ReturnType<typeof array>).toArray() as Complex[];
        // First: cond is (1,0) which is truthy -> x[0] = (10,10)
        expect(values[0].re).toBe(10);
        expect(values[0].im).toBe(10);
        // Second: cond is (0,0) which is falsy -> y[1] = (200,200)
        expect(values[1].re).toBe(200);
        expect(values[1].im).toBe(200);
        // Third: cond is (0,1) which is truthy -> x[2] = (30,30)
        expect(values[2].re).toBe(30);
        expect(values[2].im).toBe(30);
      });
    });

    describe('extract()', () => {
      it('extracts complex values where condition is true', () => {
        const cond = array([1, 0, 1, 0]);
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8)]);
        const result = extract(cond, a);

        const values = result.toArray() as Complex[];
        expect(values.length).toBe(2);
        expect(values[0].re).toBe(1);
        expect(values[0].im).toBe(2);
        expect(values[1].re).toBe(5);
        expect(values[1].im).toBe(6);
      });

      it('extracts using complex condition', () => {
        const cond = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1)]);
        const a = array([new Complex(10, 20), new Complex(30, 40), new Complex(50, 60)]);
        const result = extract(cond, a);

        const values = result.toArray() as Complex[];
        expect(values.length).toBe(2);
        expect(values[0].re).toBe(10);
        expect(values[0].im).toBe(20);
        expect(values[1].re).toBe(50);
        expect(values[1].im).toBe(60);
      });
    });

    describe('count_nonzero()', () => {
      it('counts non-zero complex numbers', () => {
        const a = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const result = count_nonzero(a);

        expect(result).toBe(2);
      });

      it('counts zero only when both parts are zero', () => {
        const a = array([new Complex(0, 0), new Complex(0, 0.001), new Complex(0.001, 0)]);
        const result = count_nonzero(a);

        expect(result).toBe(2);
      });

      it('returns 0 for all-zero array', () => {
        const a = array([new Complex(0, 0), new Complex(0, 0)]);
        const result = count_nonzero(a);

        expect(result).toBe(0);
      });
    });
  });

  describe('Logical Operations', () => {
    describe('logical_and()', () => {
      it('computes logical AND of complex arrays', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const b = array([new Complex(1, 0), new Complex(1, 0), new Complex(1, 0), new Complex(0, 0)]);
        const result = logical_and(a, b);

        expect(result.toArray()).toEqual([1, 0, 1, 0]);
      });

      it('computes logical AND with scalar', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1)]);
        const result = logical_and(a, 1);

        expect(result.toArray()).toEqual([1, 0, 1]);
      });

      it('treats complex as truthy if either part is non-zero', () => {
        const a = array([new Complex(0, 1), new Complex(1, 0), new Complex(0.5, 0.5)]);
        const b = array([new Complex(1, 0), new Complex(0, 1), new Complex(0, 0)]);
        const result = logical_and(a, b);

        expect(result.toArray()).toEqual([1, 1, 0]);
      });
    });

    describe('logical_or()', () => {
      it('computes logical OR of complex arrays', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const b = array([new Complex(0, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)]);
        const result = logical_or(a, b);

        expect(result.toArray()).toEqual([1, 1, 1, 0]);
      });

      it('computes logical OR with scalar', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1)]);
        const result = logical_or(a, 0);

        expect(result.toArray()).toEqual([1, 0, 1]);
      });
    });

    describe('logical_not()', () => {
      it('computes logical NOT of complex array', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const result = logical_not(a);

        expect(result.toArray()).toEqual([0, 1, 0, 1]);
      });

      it('returns true only when both parts are zero', () => {
        const a = array([new Complex(0, 0), new Complex(0.001, 0), new Complex(0, 0.001)]);
        const result = logical_not(a);

        expect(result.toArray()).toEqual([1, 0, 0]);
      });
    });

    describe('logical_xor()', () => {
      it('computes logical XOR of complex arrays', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1), new Complex(0, 0)]);
        const b = array([new Complex(1, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)]);
        const result = logical_xor(a, b);

        // XOR: true if exactly one is truthy
        expect(result.toArray()).toEqual([0, 1, 1, 0]);
      });

      it('computes logical XOR with scalar', () => {
        const a = array([new Complex(1, 0), new Complex(0, 0), new Complex(0, 1)]);
        const result = logical_xor(a, 1);

        expect(result.toArray()).toEqual([0, 1, 0]);
      });
    });
  });

  describe('Difference Operations', () => {
    describe('diff()', () => {
      it('computes first difference of complex array', () => {
        const a = array([new Complex(1, 2), new Complex(4, 5), new Complex(10, 8)]);
        const result = diff(a);

        const values = result.toArray() as Complex[];
        expect(values.length).toBe(2);
        // (4,5) - (1,2) = (3,3)
        expect(values[0].re).toBe(3);
        expect(values[0].im).toBe(3);
        // (10,8) - (4,5) = (6,3)
        expect(values[1].re).toBe(6);
        expect(values[1].im).toBe(3);
      });

      it('computes second difference of complex array', () => {
        const a = array([new Complex(1, 0), new Complex(2, 1), new Complex(4, 3), new Complex(7, 6)]);
        const result = diff(a, 2);

        const values = result.toArray() as Complex[];
        expect(values.length).toBe(2);
        // First diff: [(1,1), (2,2), (3,3)]
        // Second diff: [(1,1), (1,1)]
        expect(values[0].re).toBe(1);
        expect(values[0].im).toBe(1);
        expect(values[1].re).toBe(1);
        expect(values[1].im).toBe(1);
      });

      it('preserves complex dtype', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = diff(a);

        expect(result.dtype).toBe('complex128');
      });
    });

    describe('ediff1d()', () => {
      it('computes differences of flattened complex array', () => {
        const a = array([new Complex(1, 1), new Complex(3, 2), new Complex(6, 5)]);
        const result = ediff1d(a);

        const values = result.toArray() as Complex[];
        expect(values.length).toBe(2);
        // (3,2) - (1,1) = (2,1)
        expect(values[0].re).toBe(2);
        expect(values[0].im).toBe(1);
        // (6,5) - (3,2) = (3,3)
        expect(values[1].re).toBe(3);
        expect(values[1].im).toBe(3);
      });

      it('preserves complex dtype', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = ediff1d(a);

        expect(result.dtype).toBe('complex128');
      });
    });
  });

  // ==========================================================================
  // Complex NaN-aware Reduction Operations
  // ==========================================================================
  describe('NaN-aware Reduction Operations', () => {
    describe('nansum()', () => {
      it('computes sum of complex array ignoring NaN values', () => {
        // Create array with NaN values
        const a = array([
          new Complex(1, 2),
          new Complex(NaN, 3), // Will be skipped
          new Complex(4, 5),
        ]);
        const result = a.nansum();

        // Only (1+2i) + (4+5i) = 5+7i
        expect((result as Complex).re).toBe(5);
        expect((result as Complex).im).toBe(7);
      });

      it('skips values where imaginary part is NaN', () => {
        const a = array([
          new Complex(1, 2),
          new Complex(3, NaN), // Will be skipped
          new Complex(4, 5),
        ]);
        const result = a.nansum();

        expect((result as Complex).re).toBe(5);
        expect((result as Complex).im).toBe(7);
      });

      it('returns 0+0i for all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = a.nansum();

        expect((result as Complex).re).toBe(0);
        expect((result as Complex).im).toBe(0);
      });
    });

    describe('nanprod()', () => {
      it('computes product of complex array ignoring NaN values', () => {
        const a = array([
          new Complex(1, 1),
          new Complex(NaN, 2), // Will be skipped
          new Complex(2, 0),
        ]);
        const result = a.nanprod();

        // Only (1+i) * (2+0i) = 2+2i
        expect((result as Complex).re).toBeCloseTo(2);
        expect((result as Complex).im).toBeCloseTo(2);
      });

      it('returns 1+0i for all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = a.nanprod();

        expect((result as Complex).re).toBe(1);
        expect((result as Complex).im).toBe(0);
      });
    });

    describe('nanmean()', () => {
      it('computes mean of complex array ignoring NaN values', () => {
        const a = array([
          new Complex(2, 4),
          new Complex(NaN, 2), // Will be skipped
          new Complex(4, 8),
        ]);
        const result = a.nanmean();

        // Mean of (2+4i), (4+8i) = (3+6i)
        expect((result as Complex).re).toBeCloseTo(3);
        expect((result as Complex).im).toBeCloseTo(6);
      });

      it('returns NaN+NaNi for all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = a.nanmean();

        expect(isNaN((result as Complex).re)).toBe(true);
        expect(isNaN((result as Complex).im)).toBe(true);
      });
    });

    describe('nancumsum()', () => {
      it('computes cumulative sum of complex array ignoring NaN values', () => {
        const a = array([
          new Complex(1, 2),
          new Complex(NaN, 3), // Will be skipped
          new Complex(4, 5),
        ]);
        const result = nancumsum(a);

        // [1+2i, 1+2i (NaN skipped), 5+7i]
        expect(result.get([0])).toEqual(new Complex(1, 2));
        expect(result.get([1])).toEqual(new Complex(1, 2)); // NaN skipped, keeps previous sum
        expect(result.get([2])).toEqual(new Complex(5, 7));
      });

      it('skips values where imaginary part is NaN', () => {
        const a = array([
          new Complex(1, 1),
          new Complex(2, NaN), // Will be skipped
          new Complex(3, 3),
        ]);
        const result = nancumsum(a);

        expect(result.get([0])).toEqual(new Complex(1, 1));
        expect(result.get([1])).toEqual(new Complex(1, 1)); // NaN skipped
        expect(result.get([2])).toEqual(new Complex(4, 4));
      });

      it('handles all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = nancumsum(a);

        expect(result.get([0])).toEqual(new Complex(0, 0));
        expect(result.get([1])).toEqual(new Complex(0, 0));
      });
    });

    describe('nancumprod()', () => {
      it('computes cumulative product of complex array ignoring NaN values', () => {
        const a = array([
          new Complex(1, 1),
          new Complex(NaN, 2), // Will be skipped (treated as 1+0i)
          new Complex(2, 0),
        ]);
        const result = nancumprod(a);

        // [1+i, 1+i (NaN skipped), (1+i)*(2+0i) = 2+2i]
        expect(result.get([0])).toEqual(new Complex(1, 1));
        expect(result.get([1])).toEqual(new Complex(1, 1)); // NaN skipped, keeps previous product
        expect((result.get([2]) as Complex).re).toBeCloseTo(2);
        expect((result.get([2]) as Complex).im).toBeCloseTo(2);
      });

      it('handles all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = nancumprod(a);

        // All treated as 1+0i
        expect(result.get([0])).toEqual(new Complex(1, 0));
        expect(result.get([1])).toEqual(new Complex(1, 0));
      });

      it('complex multiplication works correctly', () => {
        // (2+3i) * (4+5i) = (2*4 - 3*5) + (2*5 + 3*4)i = -7 + 22i
        const a = array([new Complex(2, 3), new Complex(4, 5)]);
        const result = nancumprod(a);

        expect(result.get([0])).toEqual(new Complex(2, 3));
        expect((result.get([1]) as Complex).re).toBeCloseTo(-7);
        expect((result.get([1]) as Complex).im).toBeCloseTo(22);
      });
    });

    describe('nanvar()', () => {
      it('computes variance of complex array ignoring NaN values', () => {
        // Array: [1+2i, NaN+3i (skip), 3+4i]
        // Mean of [1+2i, 3+4i] = (2+3i)
        // Var = (|1+2i - 2-3i|² + |3+4i - 2-3i|²) / 2
        //     = (|-1-i|² + |1+i|²) / 2 = (2 + 2) / 2 = 2
        const a = array([
          new Complex(1, 2),
          new Complex(NaN, 3), // Will be skipped
          new Complex(3, 4),
        ]);
        const result = a.nanvar();

        expect(result).toBeCloseTo(2);
      });

      it('returns NaN for all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = a.nanvar();

        expect(isNaN(result as number)).toBe(true);
      });

      it('computes variance with ddof', () => {
        // Same as above but with ddof=1
        // Var = (2 + 2) / (2 - 1) = 4
        const a = array([
          new Complex(1, 2),
          new Complex(NaN, 3), // Will be skipped
          new Complex(3, 4),
        ]);
        const result = a.nanvar(undefined, 1);

        expect(result).toBeCloseTo(4);
      });
    });

    describe('nanstd()', () => {
      it('computes standard deviation of complex array ignoring NaN values', () => {
        // Same setup as nanvar - std = sqrt(var) = sqrt(2)
        const a = array([
          new Complex(1, 2),
          new Complex(NaN, 3), // Will be skipped
          new Complex(3, 4),
        ]);
        const result = a.nanstd();

        expect(result).toBeCloseTo(Math.sqrt(2));
      });

      it('returns NaN for all-NaN array', () => {
        const a = array([new Complex(NaN, 1), new Complex(2, NaN)]);
        const result = a.nanstd();

        expect(isNaN(result as number)).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Complex argmin/argmax Operations
  // ==========================================================================
  describe('Argmin/Argmax Operations', () => {
    describe('argmin()', () => {
      it('finds index of minimum complex value using lexicographic ordering', () => {
        // Lexicographic: compare real parts first, then imaginary
        const a = array([new Complex(2, 1), new Complex(1, 3), new Complex(1, 2), new Complex(3, 0)]);
        const result = a.argmin();

        // 1+2i is smallest (real=1 is smallest, and among real=1, imag=2 < imag=3)
        expect(result).toBe(2);
      });

      it('finds argmin when real parts are equal', () => {
        const a = array([new Complex(1, 5), new Complex(1, 2), new Complex(1, 3)]);
        const result = a.argmin();

        // All real parts equal, so compare imaginary: 2 is smallest
        expect(result).toBe(1);
      });

      it('finds argmin with negative real parts', () => {
        const a = array([new Complex(0, 1), new Complex(-1, 5), new Complex(1, 0)]);
        const result = a.argmin();

        // -1+5i has smallest real part
        expect(result).toBe(1);
      });

      it('finds argmin along axis', () => {
        const a = array([
          [new Complex(2, 1), new Complex(1, 0)],
          [new Complex(0, 5), new Complex(3, 2)],
        ]);
        const result = a.argmin(0);

        // Along axis 0: compare [2+1i, 0+5i] and [1+0i, 3+2i]
        // Column 0: 0+5i < 2+1i, so index 1
        // Column 1: 1+0i < 3+2i, so index 0
        expect(result.toArray()).toEqual([1, 0]);
      });
    });

    describe('argmax()', () => {
      it('finds index of maximum complex value using lexicographic ordering', () => {
        const a = array([new Complex(2, 1), new Complex(1, 3), new Complex(1, 2), new Complex(3, 0)]);
        const result = a.argmax();

        // 3+0i is largest (real=3 is largest)
        expect(result).toBe(3);
      });

      it('finds argmax when real parts are equal', () => {
        const a = array([new Complex(1, 2), new Complex(1, 5), new Complex(1, 3)]);
        const result = a.argmax();

        // All real parts equal, so compare imaginary: 5 is largest
        expect(result).toBe(1);
      });

      it('finds argmax with mixed sign values', () => {
        const a = array([new Complex(-2, 0), new Complex(1, -5), new Complex(0, 10)]);
        const result = a.argmax();

        // 1-5i has largest real part
        expect(result).toBe(1);
      });

      it('finds argmax along axis', () => {
        const a = array([
          [new Complex(2, 1), new Complex(1, 0)],
          [new Complex(0, 5), new Complex(3, 2)],
        ]);
        const result = a.argmax(0);

        // Along axis 0: compare [2+1i, 0+5i] and [1+0i, 3+2i]
        // Column 0: 2+1i > 0+5i, so index 0
        // Column 1: 3+2i > 1+0i, so index 1
        expect(result.toArray()).toEqual([0, 1]);
      });
    });
  });

  // ==========================================================================
  // Complex Gradient and Cross Operations
  // ==========================================================================
  describe('Gradient and Cross Operations', () => {
    describe('gradient()', () => {
      it('computes gradient of complex array', () => {
        // f = [1+1i, 2+3i, 4+6i, 7+10i]
        // gradient uses central differences in interior, forward/backward at edges
        const f = array([new Complex(1, 1), new Complex(2, 3), new Complex(4, 6), new Complex(7, 10)]);
        const result = gradient(f);

        expect(result.dtype).toBe('complex128');
        const values = (result as typeof f).toArray() as Complex[];
        expect(values.length).toBe(4);

        // First element: forward difference = (2+3i) - (1+1i) = 1+2i
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(2);

        // Second element: central difference = ((4+6i) - (1+1i)) / 2 = 1.5+2.5i
        expect(values[1].re).toBeCloseTo(1.5);
        expect(values[1].im).toBeCloseTo(2.5);

        // Third element: central difference = ((7+10i) - (2+3i)) / 2 = 2.5+3.5i
        expect(values[2].re).toBeCloseTo(2.5);
        expect(values[2].im).toBeCloseTo(3.5);

        // Fourth element: backward difference = (7+10i) - (4+6i) = 3+4i
        expect(values[3].re).toBeCloseTo(3);
        expect(values[3].im).toBeCloseTo(4);
      });

      it('computes gradient with custom spacing', () => {
        const f = array([new Complex(0, 0), new Complex(2, 4), new Complex(6, 12)]);
        const result = gradient(f, 2);

        expect(result.dtype).toBe('complex128');
        const values = (result as typeof f).toArray() as Complex[];

        // First: forward difference / 2 = (2+4i - 0) / 2 = 1+2i
        expect(values[0].re).toBeCloseTo(1);
        expect(values[0].im).toBeCloseTo(2);

        // Second: central difference / (2*2) = (6+12i - 0) / 4 = 1.5+3i
        expect(values[1].re).toBeCloseTo(1.5);
        expect(values[1].im).toBeCloseTo(3);

        // Third: backward difference / 2 = (6+12i - 2+4i) / 2 = 2+4i
        expect(values[2].re).toBeCloseTo(2);
        expect(values[2].im).toBeCloseTo(4);
      });

      it('preserves complex dtype', () => {
        const f = array([new Complex(1, 2), new Complex(3, 4)]);
        const result = gradient(f);

        expect(result.dtype).toBe('complex128');
      });
    });

    describe('cross()', () => {
      it('computes cross product of complex 3D vectors', () => {
        // a = [1+1i, 2+0i, 0+1i]
        // b = [0+1i, 1+0i, 2+0i]
        const a = array([new Complex(1, 1), new Complex(2, 0), new Complex(0, 1)]);
        const b = array([new Complex(0, 1), new Complex(1, 0), new Complex(2, 0)]);
        const result = cross(a, b);

        expect(result.dtype).toBe('complex128');
        expect(result.shape).toEqual([3]);

        const values = result.toArray() as Complex[];

        // c0 = a1*b2 - a2*b1 = (2+0i)*(2+0i) - (0+1i)*(1+0i) = 4 - i = 4-i
        expect(values[0].re).toBeCloseTo(4);
        expect(values[0].im).toBeCloseTo(-1);

        // c1 = a2*b0 - a0*b2 = (0+1i)*(0+1i) - (1+1i)*(2+0i) = -1 - 2 - 2i = -3-2i
        expect(values[1].re).toBeCloseTo(-3);
        expect(values[1].im).toBeCloseTo(-2);

        // c2 = a0*b1 - a1*b0 = (1+1i)*(1+0i) - (2+0i)*(0+1i) = 1+i - 2i = 1-i
        expect(values[2].re).toBeCloseTo(1);
        expect(values[2].im).toBeCloseTo(-1);
      });

      it('computes cross product of complex 2D vectors (returns scalar)', () => {
        // a = [1+1i, 2+0i]
        // b = [0+1i, 1+0i]
        const a = array([new Complex(1, 1), new Complex(2, 0)]);
        const b = array([new Complex(0, 1), new Complex(1, 0)]);
        const result = cross(a, b);

        expect(result.dtype).toBe('complex128');
        expect(result.shape).toEqual([]);

        const values = result.toArray() as Complex;

        // c = a0*b1 - a1*b0 = (1+1i)*(1+0i) - (2+0i)*(0+1i) = 1+i - 2i = 1-i
        expect(values.re).toBeCloseTo(1);
        expect(values.im).toBeCloseTo(-1);
      });

      it('computes cross product with purely imaginary vectors', () => {
        // a = [i, 2i, 3i]
        // b = [1, 2, 3]
        const a = array([new Complex(0, 1), new Complex(0, 2), new Complex(0, 3)]);
        const b = array([new Complex(1, 0), new Complex(2, 0), new Complex(3, 0)]);
        const result = cross(a, b);

        const values = result.toArray() as Complex[];

        // c0 = (2i)*(3) - (3i)*(2) = 6i - 6i = 0
        expect(values[0].re).toBeCloseTo(0);
        expect(values[0].im).toBeCloseTo(0);

        // c1 = (3i)*(1) - (i)*(3) = 3i - 3i = 0
        expect(values[1].re).toBeCloseTo(0);
        expect(values[1].im).toBeCloseTo(0);

        // c2 = (i)*(2) - (2i)*(1) = 2i - 2i = 0
        expect(values[2].re).toBeCloseTo(0);
        expect(values[2].im).toBeCloseTo(0);
      });

      it('preserves complex dtype', () => {
        const a = array([new Complex(1, 0), new Complex(0, 1), new Complex(1, 1)]);
        const b = array([new Complex(0, 1), new Complex(1, 0), new Complex(0, 0)]);
        const result = cross(a, b);

        expect(result.dtype).toBe('complex128');
      });
    });
  });
});
