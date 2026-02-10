/**
 * Unit tests for division operations
 * Tests division behavior including type promotion and division by zero
 */

import { describe, it, expect } from 'vitest';
import { ones, zeros, array } from '../../src';
import type { DType } from '../../src/common/dtype';

describe('Division Operations', () => {
  describe('Integer division promotes to float64 (NumPy behavior)', () => {
    const intTypes: DType[] = [
      'int8',
      'int16',
      'int32',
      'int64',
      'uint8',
      'uint16',
      'uint32',
      'uint64',
    ];

    for (const dtype of intTypes) {
      it(`${dtype} / scalar returns float64`, () => {
        const arr = array([4, 6, 8], dtype);
        const result = arr.divide(2);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(2);
        expect(result.get([1])).toBe(3);
        expect(result.get([2])).toBe(4);
      });

      it(`${dtype} array / ${dtype} array returns float64`, () => {
        const a = array([4, 6, 8], dtype);
        const b = array([2, 2, 2], dtype);
        const result = a.divide(b);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(2);
        expect(result.get([1])).toBe(3);
        expect(result.get([2])).toBe(4);
      });
    }
  });

  describe('Float division preserves dtype', () => {
    it('float32 / scalar returns float32', () => {
      const arr = array([4, 6, 8], 'float32');
      const result = arr.divide(2);

      expect(result.dtype).toBe('float32');
      expect(result.get([0])).toBe(2);
    });

    it('float64 / scalar returns float64', () => {
      const arr = array([4, 6, 8], 'float64');
      const result = arr.divide(2);

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(2);
    });

    it('float32 array / float32 array returns float32', () => {
      const a = array([4, 6, 8], 'float32');
      const b = array([2, 2, 2], 'float32');
      const result = a.divide(b);

      expect(result.dtype).toBe('float32');
      expect(result.get([0])).toBe(2);
    });
  });

  describe('Division by zero returns Infinity/NaN (NumPy behavior)', () => {
    describe('Scalar division by zero', () => {
      it('float64: positive / 0 returns Infinity', () => {
        const arr = ones([3], 'float64');
        const result = arr.divide(0);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(Infinity);
        expect(result.get([1])).toBe(Infinity);
        expect(result.get([2])).toBe(Infinity);
      });

      it('float64: negative / 0 returns -Infinity', () => {
        const arr = ones([2], 'float64').multiply(-1);
        const result = arr.divide(0);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(-Infinity);
        expect(result.get([1])).toBe(-Infinity);
      });

      it('float64: 0 / 0 returns NaN', () => {
        const arr = zeros([2], 'float64');
        const result = arr.divide(0);

        expect(result.dtype).toBe('float64');
        expect(Number.isNaN(result.get([0]))).toBe(true);
        expect(Number.isNaN(result.get([1]))).toBe(true);
      });

      it('float32: positive / 0 returns Infinity', () => {
        const arr = ones([2], 'float32');
        const result = arr.divide(0);

        expect(result.dtype).toBe('float32');
        expect(result.get([0])).toBe(Infinity);
      });

      it('float32: 0 / 0 returns NaN', () => {
        const arr = zeros([2], 'float32');
        const result = arr.divide(0);

        expect(result.dtype).toBe('float32');
        expect(Number.isNaN(result.get([0]))).toBe(true);
      });

      const intTypes: DType[] = [
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
      ];

      for (const dtype of intTypes) {
        it(`${dtype}: ones / 0 returns Infinity (promotes to float64)`, () => {
          const arr = ones([2], dtype);
          const result = arr.divide(0);

          expect(result.dtype).toBe('float64');
          expect(result.get([0])).toBe(Infinity);
          expect(result.get([1])).toBe(Infinity);
        });
      }
    });

    describe('Array-to-array division by zero', () => {
      it('float64 array divided by array with zeros', () => {
        const a = array([1, 2, 0, -1, 5], 'float64');
        const b = array([1, 0, 0, 0, 2], 'float64');
        const result = a.divide(b);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(1); // 1 / 1 = 1
        expect(result.get([1])).toBe(Infinity); // 2 / 0 = Inf
        expect(Number.isNaN(result.get([2]))).toBe(true); // 0 / 0 = NaN
        expect(result.get([3])).toBe(-Infinity); // -1 / 0 = -Inf
        expect(result.get([4])).toBe(2.5); // 5 / 2 = 2.5
      });

      it('int32 array divided by array with zeros (promotes to float64)', () => {
        const a = array([1, 2, 3], 'int32');
        const b = array([1, 0, 1], 'int32');
        const result = a.divide(b);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(1); // 1 / 1 = 1
        expect(result.get([1])).toBe(Infinity); // 2 / 0 = Inf
        expect(result.get([2])).toBe(3); // 3 / 1 = 3
      });

      it('int64 array divided by array with zeros (promotes to float64)', () => {
        const a = ones([3], 'int64');
        const b = zeros([3], 'int64');
        const result = a.divide(b);

        expect(result.dtype).toBe('float64');
        expect(result.get([0])).toBe(Infinity);
        expect(result.get([1])).toBe(Infinity);
        expect(result.get([2])).toBe(Infinity);
      });
    });
  });

  describe('Mixed scenarios', () => {
    it('float64: mix of normal division and division by zero', () => {
      const a = array([10, 20, 30, 0, -15], 'float64');
      const b = array([2, 0, 5, 0, 3], 'float64');
      const result = a.divide(b);

      expect(result.get([0])).toBe(5); // 10 / 2 = 5
      expect(result.get([1])).toBe(Infinity); // 20 / 0 = Inf
      expect(result.get([2])).toBe(6); // 30 / 5 = 6
      expect(Number.isNaN(result.get([3]))).toBe(true); // 0 / 0 = NaN
      expect(result.get([4])).toBe(-5); // -15 / 3 = -5
    });

    it('int32 mixed with normal division and division by zero', () => {
      const a = array([10, 20, 30], 'int32');
      const b = array([2, 0, 5], 'int32');
      const result = a.divide(b);

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(5); // 10 / 2 = 5
      expect(result.get([1])).toBe(Infinity); // 20 / 0 = Inf
      expect(result.get([2])).toBe(6); // 30 / 5 = 6
    });
  });

  describe('Edge cases', () => {
    it('float64: very small number divided by zero', () => {
      const arr = array([1e-308, -1e-308], 'float64');
      const result = arr.divide(0);

      expect(result.get([0])).toBe(Infinity);
      expect(result.get([1])).toBe(-Infinity);
    });

    it('float64: large number divided by zero', () => {
      const arr = array([1e308, -1e308], 'float64');
      const result = arr.divide(0);

      expect(result.get([0])).toBe(Infinity);
      expect(result.get([1])).toBe(-Infinity);
    });

    it('bool dtype division by zero (promotes to float64)', () => {
      const arr = array([1, 0, 1], 'bool');
      const result = arr.divide(0);

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(Infinity); // true / 0 = Inf
      expect(Number.isNaN(result.get([1]))).toBe(true); // false / 0 = NaN
    });

    it('2D array division by zero', () => {
      const arr = ones([2, 3], 'int32');
      const result = arr.divide(0);

      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([2, 3]);
      expect(result.get([0, 0])).toBe(Infinity);
      expect(result.get([1, 2])).toBe(Infinity);
    });
  });

  describe('Broadcasting with division', () => {
    it('broadcasts correctly with division by zero', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int32'
      );
      const b = array([1, 0], 'int32');
      const result = a.divide(b);

      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([2, 2]);
      expect(result.get([0, 0])).toBe(1); // 1 / 1 = 1
      expect(result.get([0, 1])).toBe(Infinity); // 2 / 0 = Inf
      expect(result.get([1, 0])).toBe(3); // 3 / 1 = 3
      expect(result.get([1, 1])).toBe(Infinity); // 4 / 0 = Inf
    });
  });
});
