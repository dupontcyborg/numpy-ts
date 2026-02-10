/**
 * Unit tests for rounding operations
 */

import { describe, it, expect } from 'vitest';
import { array, around, ceil, fix, floor, rint, round, trunc } from '../../src';

describe('Rounding Functions', () => {
  describe('around()', () => {
    it("rounds to 0 decimals by default (banker's rounding)", () => {
      // NumPy uses banker's rounding (round half to even)
      // 1.5 -> 2 (round to even), 2.5 -> 2 (round to even), -1.5 -> -2 (round to even)
      const arr = array([1.4, 1.5, 1.6, 2.5, -1.5]);
      const result = around(arr);
      expect(result.toArray()).toEqual([1, 2, 2, 2, -2]);
    });

    it('rounds to specified decimals', () => {
      const arr = array([1.234, 2.567, 3.891]);
      const result = around(arr, 2);
      expect(result.toArray()).toEqual([1.23, 2.57, 3.89]);
    });

    it('rounds to negative decimals', () => {
      const arr = array([123, 456, 789]);
      const result = around(arr, -1);
      expect(result.toArray()).toEqual([120, 460, 790]);
    });

    it('handles 2D array', () => {
      const arr = array([
        [1.4, 2.5],
        [3.6, 4.1],
      ]);
      const result = around(arr);
      // 2.5 rounds to 2 (banker's rounding - round to even)
      expect(result.toArray()).toEqual([
        [1, 2],
        [4, 4],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1.4, 1.5, 1.6]);
      const result = arr.around();
      expect(result.toArray()).toEqual([1, 2, 2]);
    });
  });

  describe('ceil()', () => {
    it('returns ceiling of elements', () => {
      const arr = array([1.1, 1.9, -1.1, -1.9]);
      const result = ceil(arr);
      expect(result.toArray()).toEqual([2, 2, -1, -1]);
    });

    it('handles 2D array', () => {
      const arr = array([
        [1.1, 2.5],
        [3.9, 4.0],
      ]);
      const result = ceil(arr);
      expect(result.toArray()).toEqual([
        [2, 3],
        [4, 4],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1.1, 1.9]);
      const result = arr.ceil();
      expect(result.toArray()).toEqual([2, 2]);
    });
  });

  describe('fix()', () => {
    it('rounds towards zero', () => {
      const arr = array([1.9, -1.9, 2.5, -2.5]);
      const result = fix(arr);
      expect(result.toArray()).toEqual([1, -1, 2, -2]);
    });

    it('handles 2D array', () => {
      const arr = array([
        [1.7, -1.7],
        [2.3, -2.3],
      ]);
      const result = fix(arr);
      expect(result.toArray()).toEqual([
        [1, -1],
        [2, -2],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1.9, -1.9]);
      const result = arr.fix();
      expect(result.toArray()).toEqual([1, -1]);
    });
  });

  describe('floor()', () => {
    it('returns floor of elements', () => {
      const arr = array([1.1, 1.9, -1.1, -1.9]);
      const result = floor(arr);
      expect(result.toArray()).toEqual([1, 1, -2, -2]);
    });

    it('handles 2D array', () => {
      const arr = array([
        [1.1, 2.5],
        [3.9, 4.0],
      ]);
      const result = floor(arr);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1.1, 1.9]);
      const result = arr.floor();
      expect(result.toArray()).toEqual([1, 1]);
    });
  });

  describe('rint()', () => {
    it("rounds to nearest integer (banker's rounding)", () => {
      // NumPy uses banker's rounding (round half to even)
      const arr = array([1.4, 1.5, 1.6, 2.5, -1.5]);
      const result = rint(arr);
      expect(result.toArray()).toEqual([1, 2, 2, 2, -2]);
    });

    it('handles 2D array', () => {
      const arr = array([
        [1.4, 2.5],
        [3.6, 4.1],
      ]);
      const result = rint(arr);
      // 2.5 rounds to 2 (banker's rounding - round to even)
      expect(result.toArray()).toEqual([
        [1, 2],
        [4, 4],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1.4, 1.6]);
      const result = arr.rint();
      expect(result.toArray()).toEqual([1, 2]);
    });
  });

  describe('round()', () => {
    it('is alias for around', () => {
      const arr = array([1.4, 1.5, 1.6]);
      const result = round(arr);
      expect(result.toArray()).toEqual([1, 2, 2]);
    });

    it('accepts decimals parameter', () => {
      const arr = array([1.234, 2.567]);
      const result = round(arr, 2);
      expect(result.toArray()).toEqual([1.23, 2.57]);
    });

    it('uses method syntax', () => {
      const arr = array([1.4, 1.5, 1.6]);
      const result = arr.round();
      expect(result.toArray()).toEqual([1, 2, 2]);
    });
  });

  describe('trunc()', () => {
    it('truncates to integer part', () => {
      const arr = array([1.7, -1.7, 2.9, -2.9]);
      const result = trunc(arr);
      expect(result.toArray()).toEqual([1, -1, 2, -2]);
    });

    it('handles 2D array', () => {
      const arr = array([
        [1.9, -1.9],
        [2.1, -2.1],
      ]);
      const result = trunc(arr);
      expect(result.toArray()).toEqual([
        [1, -1],
        [2, -2],
      ]);
    });

    it('uses method syntax', () => {
      const arr = array([1.7, -1.7]);
      const result = arr.trunc();
      expect(result.toArray()).toEqual([1, -1]);
    });
  });
});

describe('Edge Cases', () => {
  it('ceil handles empty array', () => {
    const arr = array([]);
    const result = ceil(arr);
    expect(result.toArray()).toEqual([]);
  });

  it('floor handles single element', () => {
    const arr = array([1.5]);
    const result = floor(arr);
    expect(result.toArray()).toEqual([1]);
  });

  it('around handles integer input', () => {
    const arr = array([1, 2, 3]);
    const result = around(arr);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it('trunc handles very small values', () => {
    const arr = array([0.001, -0.001]);
    const result = trunc(arr);
    const resultArr = result.toArray() as number[];
    expect(resultArr[0]).toBe(0);
    // Math.trunc(-0.001) returns -0, which is numerically equivalent to 0
    expect(Object.is(resultArr[1], 0) || Object.is(resultArr[1], -0)).toBe(true);
  });

  it('round handles 3D array', () => {
    const arr = array([
      [
        [1.1, 2.9],
        [3.5, 4.4],
      ],
    ]);
    const result = round(arr);
    expect(result.toArray()).toEqual([
      [
        [1, 3],
        [4, 4],
      ],
    ]);
  });
});
