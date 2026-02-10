import { describe, it, expect } from 'vitest';
import { array, exp, exp2, expm1, log, log2, log10, log1p, logaddexp, logaddexp2 } from '../../src';

describe('Exponential Operations', () => {
  describe('exp', () => {
    it('computes e^x for each element', () => {
      const arr = array([0, 1, 2]);
      const result = exp(arr);
      const expected = [1, Math.E, Math.E ** 2];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.exp();
      expect(Math.abs(result.get([0]) - 1)).toBeLessThan(1e-10);
      expect(Math.abs(result.get([1]) - Math.E)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = exp(arr);
      expect(result.dtype).toBe('float64');
    });

    it('preserves float32 dtype', () => {
      const arr = array([0, 1], 'float32');
      const result = exp(arr);
      expect(result.dtype).toBe('float32');
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [0, 1],
        [2, 3],
      ]);
      const result = exp(arr);
      expect(result.shape).toEqual([2, 2]);
    });

    it('satisfies exp(0) = 1', () => {
      const arr = array([0]);
      const result = exp(arr);
      expect(result.get([0])).toBe(1);
    });
  });

  describe('exp2', () => {
    it('computes 2^x for each element', () => {
      const arr = array([0, 1, 2, 3]);
      const result = exp2(arr);
      const expected = [1, 2, 4, 8];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 3]);
      const result = arr.exp2();
      expect(result.get([0])).toBe(1);
      expect(result.get([1])).toBe(8);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = exp2(arr);
      expect(result.dtype).toBe('float64');
    });

    it('handles negative exponents', () => {
      const arr = array([-1, -2]);
      const result = exp2(arr);
      expect(Math.abs(result.get([0]) - 0.5)).toBeLessThan(1e-10);
      expect(Math.abs(result.get([1]) - 0.25)).toBeLessThan(1e-10);
    });
  });

  describe('expm1', () => {
    it('computes e^x - 1 for each element', () => {
      const arr = array([0, 1, 2]);
      const result = expm1(arr);
      const expected = [0, Math.E - 1, Math.E ** 2 - 1];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('is more accurate than exp(x) - 1 for small x', () => {
      // For very small x, expm1 is more accurate
      const smallX = array([1e-15]);
      const result = expm1(smallX);
      // expm1(1e-15) should be approximately 1e-15
      expect(Math.abs(result.get([0]) - 1e-15)).toBeLessThan(1e-20);
    });

    it('satisfies expm1(0) = 0', () => {
      const arr = array([0]);
      const result = expm1(arr);
      expect(result.get([0])).toBe(0);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1], 'int32');
      const result = expm1(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('log', () => {
    it('computes natural logarithm', () => {
      const arr = array([1, Math.E, Math.E ** 2]);
      const result = log(arr);
      const expected = [0, 1, 2];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([1, Math.E]);
      const result = arr.log();
      expect(Math.abs(result.get([0]))).toBeLessThan(1e-10);
      expect(Math.abs(result.get([1]) - 1)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = log(arr);
      expect(result.dtype).toBe('float64');
    });

    it('returns -Infinity for log(0)', () => {
      const arr = array([0]);
      const result = log(arr);
      expect(result.get([0])).toBe(-Infinity);
    });

    it('returns NaN for negative values', () => {
      const arr = array([-1]);
      const result = log(arr);
      expect(Number.isNaN(result.get([0]))).toBe(true);
    });
  });

  describe('log2', () => {
    it('computes base-2 logarithm', () => {
      const arr = array([1, 2, 4, 8]);
      const result = log2(arr);
      const expected = [0, 1, 2, 3];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([1, 8]);
      const result = arr.log2();
      expect(result.get([0])).toBe(0);
      expect(result.get([1])).toBe(3);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([1, 2], 'int32');
      const result = log2(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('log10', () => {
    it('computes base-10 logarithm', () => {
      const arr = array([1, 10, 100, 1000]);
      const result = log10(arr);
      const expected = [0, 1, 2, 3];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([1, 100]);
      const result = arr.log10();
      expect(result.get([0])).toBe(0);
      expect(result.get([1])).toBe(2);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([1, 10], 'int32');
      const result = log10(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('log1p', () => {
    it('computes log(1 + x)', () => {
      const arr = array([0, Math.E - 1, Math.E ** 2 - 1]);
      const result = log1p(arr);
      const expected = [0, 1, 2];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('is more accurate than log(1 + x) for small x', () => {
      const smallX = array([1e-15]);
      const result = log1p(smallX);
      // log1p(1e-15) should be approximately 1e-15
      expect(Math.abs(result.get([0]) - 1e-15)).toBeLessThan(1e-20);
    });

    it('satisfies log1p(0) = 0', () => {
      const arr = array([0]);
      const result = log1p(arr);
      expect(result.get([0])).toBe(0);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1], 'int32');
      const result = log1p(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('logaddexp', () => {
    it('computes log(exp(x1) + exp(x2))', () => {
      const x1 = array([1, 2, 3]);
      const x2 = array([1, 2, 3]);
      const result = logaddexp(x1, x2);
      // log(exp(x) + exp(x)) = log(2*exp(x)) = x + log(2)
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < 3; i++) {
        const expected = i + 1 + Math.log(2);
        expect(Math.abs(resultArr[i]! - expected)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const x1 = array([0, 1]);
      const x2 = array([0, 1]);
      const result = x1.logaddexp(x2);
      expect(Math.abs(result.get([0]) - Math.log(2))).toBeLessThan(1e-10);
    });

    it('works with scalar', () => {
      const x1 = array([1, 2]);
      const result = logaddexp(x1, 0);
      const resultArr = result.toArray() as number[];
      // log(exp(x1) + exp(0)) = log(exp(x1) + 1)
      for (let i = 0; i < 2; i++) {
        const expected = Math.log(Math.exp(i + 1) + 1);
        expect(Math.abs(resultArr[i]! - expected)).toBeLessThan(1e-10);
      }
    });

    it('is numerically stable for large values', () => {
      // Without stability, log(exp(1000) + exp(1000)) would overflow
      const x1 = array([1000]);
      const x2 = array([1000]);
      const result = logaddexp(x1, x2);
      const expected = 1000 + Math.log(2);
      expect(Math.abs(result.get([0]) - expected)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const x1 = array([1, 2], 'int32');
      const x2 = array([1, 2], 'int32');
      const result = logaddexp(x1, x2);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('logaddexp2', () => {
    it('computes log2(2^x1 + 2^x2)', () => {
      const x1 = array([1, 2, 3]);
      const x2 = array([1, 2, 3]);
      const result = logaddexp2(x1, x2);
      // log2(2^x + 2^x) = log2(2*2^x) = x + 1
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < 3; i++) {
        const expected = i + 1 + 1;
        expect(Math.abs(resultArr[i]! - expected)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const x1 = array([0, 1]);
      const x2 = array([0, 1]);
      const result = x1.logaddexp2(x2);
      expect(Math.abs(result.get([0]) - 1)).toBeLessThan(1e-10); // log2(2) = 1
      expect(Math.abs(result.get([1]) - 2)).toBeLessThan(1e-10); // log2(4) = 2
    });

    it('works with scalar', () => {
      const x1 = array([1, 2]);
      const result = logaddexp2(x1, 0);
      const resultArr = result.toArray() as number[];
      // log2(2^x1 + 2^0) = log2(2^x1 + 1)
      for (let i = 0; i < 2; i++) {
        const expected = Math.log2(Math.pow(2, i + 1) + 1);
        expect(Math.abs(resultArr[i]! - expected)).toBeLessThan(1e-10);
      }
    });

    it('is numerically stable for large values', () => {
      const x1 = array([1000]);
      const x2 = array([1000]);
      const result = logaddexp2(x1, x2);
      const expected = 1000 + 1; // log2(2^1000 + 2^1000) = log2(2*2^1000) = 1001
      expect(Math.abs(result.get([0]) - expected)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const x1 = array([1, 2], 'int32');
      const x2 = array([1, 2], 'int32');
      const result = logaddexp2(x1, x2);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('exp/log roundtrip', () => {
    it('log(exp(x)) = x', () => {
      const original = array([-2, -1, 0, 1, 2]);
      const expVal = exp(original);
      const backToOriginal = log(expVal);
      const resultArr = backToOriginal.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });

    it('exp2(log2(x)) = x for positive x', () => {
      const original = array([0.5, 1, 2, 4, 8]);
      const log2Val = log2(original);
      const backToOriginal = exp2(log2Val);
      const resultArr = backToOriginal.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });
  });
});
