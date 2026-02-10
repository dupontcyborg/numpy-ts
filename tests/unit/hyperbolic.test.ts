import { describe, it, expect } from 'vitest';
import { array, sinh, cosh, tanh, arcsinh, arccosh, arctanh } from '../../src';
import { Complex } from '../../src/common/complex';

describe('Hyperbolic Operations', () => {
  describe('sinh', () => {
    it('computes hyperbolic sine', () => {
      const arr = array([0, 1, -1]);
      const result = sinh(arr);
      const expected = [0, Math.sinh(1), Math.sinh(-1)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.sinh();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.sinh(1))).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = sinh(arr);
      expect(result.dtype).toBe('float64');
    });

    it('preserves float32 dtype', () => {
      const arr = array([0, 1], 'float32');
      const result = sinh(arr);
      expect(result.dtype).toBe('float32');
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [0, 1],
        [2, 3],
      ]);
      const result = sinh(arr);
      expect(result.shape).toEqual([2, 2]);
      const resultArr = result.toArray() as number[][];
      expect(Math.abs(resultArr[0]![0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[0]![1]! - Math.sinh(1))).toBeLessThan(1e-10);
    });

    it('satisfies sinh(0) = 0', () => {
      const arr = array([0]);
      const result = sinh(arr);
      expect(result.get([0])).toBe(0);
    });
  });

  describe('cosh', () => {
    it('computes hyperbolic cosine', () => {
      const arr = array([0, 1, -1]);
      const result = cosh(arr);
      const expected = [1, Math.cosh(1), Math.cosh(-1)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.cosh();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.cosh(1))).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = cosh(arr);
      expect(result.dtype).toBe('float64');
    });

    it('satisfies cosh(0) = 1', () => {
      const arr = array([0]);
      const result = cosh(arr);
      expect(result.get([0])).toBe(1);
    });

    it('cosh is even function: cosh(-x) = cosh(x)', () => {
      const x = array([1, 2, 3]);
      const negX = array([-1, -2, -3]);
      const coshX = cosh(x).toArray() as number[];
      const coshNegX = cosh(negX).toArray() as number[];
      for (let i = 0; i < coshX.length; i++) {
        expect(Math.abs(coshX[i]! - coshNegX[i]!)).toBeLessThan(1e-10);
      }
    });
  });

  describe('tanh', () => {
    it('computes hyperbolic tangent', () => {
      const arr = array([0, 1, -1]);
      const result = tanh(arr);
      const expected = [0, Math.tanh(1), Math.tanh(-1)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.tanh();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.tanh(1))).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = tanh(arr);
      expect(result.dtype).toBe('float64');
    });

    it('satisfies tanh(0) = 0', () => {
      const arr = array([0]);
      const result = tanh(arr);
      expect(result.get([0])).toBe(0);
    });

    it('output is in range [-1, 1]', () => {
      const arr = array([-100, -10, -1, 0, 1, 10, 100]);
      const result = tanh(arr);
      const resultArr = result.toArray() as number[];
      for (const val of resultArr) {
        // Due to floating-point precision, tanh of large values may equal exactly ±1
        expect(val).toBeGreaterThanOrEqual(-1);
        expect(val).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('arcsinh', () => {
    it('computes inverse hyperbolic sine', () => {
      const arr = array([0, 1, -1]);
      const result = arcsinh(arr);
      const expected = [0, Math.asinh(1), Math.asinh(-1)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.arcsinh();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.asinh(1))).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = arcsinh(arr);
      expect(result.dtype).toBe('float64');
    });

    it('roundtrip with sinh', () => {
      const original = array([-2, -1, 0, 1, 2]);
      const sinhVal = sinh(original);
      const backToOriginal = arcsinh(sinhVal);
      const resultArr = backToOriginal.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });
  });

  describe('arccosh', () => {
    it('computes inverse hyperbolic cosine', () => {
      const arr = array([1, 2, 10]);
      const result = arccosh(arr);
      const expected = [0, Math.acosh(2), Math.acosh(10)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([1, 2]);
      const result = arr.arccosh();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.acosh(2))).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = arccosh(arr);
      expect(result.dtype).toBe('float64');
    });

    it('returns NaN for values < 1', () => {
      const arr = array([0, 0.5]);
      const result = arccosh(arr);
      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(Number.isNaN(result.get([1]))).toBe(true);
    });

    it('roundtrip with cosh for positive values', () => {
      const original = array([0, 1, 2, 3]);
      const coshVal = cosh(original);
      const backToOriginal = arccosh(coshVal);
      const resultArr = backToOriginal.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });

    it('handles complex numbers with Re < 1 (branch cut adjustment)', () => {
      // Complex acosh with purely real z < 1 triggers the branch cut adjustment
      const arr = array([new Complex(0.5, 0)]);
      const result = arccosh(arr);

      expect(result.dtype).toBe('complex128');
      const val = result.get([0]) as any;
      // acosh(0.5) has imaginary component due to branch cut
      expect(typeof val.re).toBe('number');
      expect(typeof val.im).toBe('number');
    });
  });

  describe('arctanh', () => {
    it('computes inverse hyperbolic tangent', () => {
      const arr = array([0, 0.5, -0.5]);
      const result = arctanh(arr);
      const expected = [0, Math.atanh(0.5), Math.atanh(-0.5)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 0.5]);
      const result = arr.arctanh();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.atanh(0.5))).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0], 'int32');
      const result = arctanh(arr);
      expect(result.dtype).toBe('float64');
    });

    it('returns Infinity at boundaries', () => {
      const arr = array([1, -1]);
      const result = arctanh(arr);
      expect(result.get([0])).toBe(Infinity);
      expect(result.get([1])).toBe(-Infinity);
    });

    it('returns NaN for values outside (-1, 1)', () => {
      const arr = array([2, -2]);
      const result = arctanh(arr);
      expect(Number.isNaN(result.get([0]))).toBe(true);
      expect(Number.isNaN(result.get([1]))).toBe(true);
    });

    it('roundtrip with tanh', () => {
      const original = array([-2, -1, 0, 1, 2]);
      const tanhVal = tanh(original);
      const backToOriginal = arctanh(tanhVal);
      const resultArr = backToOriginal.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });
  });
});
