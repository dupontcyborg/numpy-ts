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
});
