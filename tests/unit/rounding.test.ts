/**
 * Unit tests for rounding operations
 */

import { describe, it, expect } from 'vitest';
import { array, around, ceil, fix, floor, rint, round, round_, trunc } from '../../src';

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

describe('Rounding - Branch Coverage Improvement', () => {
  describe('fix() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.9, -1.9, 2.5, -2.5], 'float32');
      const result = fix(arr);
      expect(result.toArray()).toEqual([1, -1, 2, -2]);
    });

    it('handles int32 dtype (no-op since already integers)', () => {
      const arr = array([1, -1, 2, -2], 'int32');
      const result = fix(arr);
      expect(result.toArray()).toEqual([1, -1, 2, -2]);
    });

    it('handles Infinity and -Infinity', () => {
      const arr = array([Infinity, -Infinity, 0]);
      const result = fix(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBe(Infinity);
      expect(values[1]).toBe(-Infinity);
      expect(values[2]).toBe(0);
    });

    it('handles NaN values', () => {
      const arr = array([NaN, 1.5, -NaN]);
      const result = fix(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBeNaN();
      expect(values[1]).toBe(1);
    });
  });

  describe('trunc() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.7, -1.7, 2.9, -2.9], 'float32');
      const result = trunc(arr);
      expect(result.toArray()).toEqual([1, -1, 2, -2]);
    });

    it('handles int32 dtype (no-op since already integers)', () => {
      const arr = array([1, -1, 2, -2], 'int32');
      const result = trunc(arr);
      expect(result.toArray()).toEqual([1, -1, 2, -2]);
    });

    it('handles Infinity and -Infinity', () => {
      const arr = array([Infinity, -Infinity]);
      const result = trunc(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBe(Infinity);
      expect(values[1]).toBe(-Infinity);
    });

    it('handles NaN values', () => {
      const arr = array([NaN, 1.5]);
      const result = trunc(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBeNaN();
      expect(values[1]).toBe(1);
    });
  });

  describe('around() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.4, 1.5, 1.6], 'float32');
      const result = around(arr);
      expect(result.toArray()).toEqual([1, 2, 2]);
    });

    it('handles int32 dtype', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = around(arr);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles large decimals', () => {
      const arr = array([1.23456789]);
      const result = around(arr, 5);
      expect(result.toArray()[0]).toBeCloseTo(1.23457, 5);
    });

    it('handles Infinity', () => {
      const arr = array([Infinity, -Infinity]);
      const result = around(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBe(Infinity);
      expect(values[1]).toBe(-Infinity);
    });

    it('handles NaN', () => {
      const arr = array([NaN]);
      const result = around(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBeNaN();
    });
  });

  describe('ceil() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.1, 1.9, -1.1, -1.9], 'float32');
      const result = ceil(arr);
      expect(result.toArray()).toEqual([2, 2, -1, -1]);
    });

    it('handles int32 dtype', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = ceil(arr);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles Infinity', () => {
      const arr = array([Infinity, -Infinity]);
      const result = ceil(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBe(Infinity);
      expect(values[1]).toBe(-Infinity);
    });
  });

  describe('floor() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.1, 1.9, -1.1, -1.9], 'float32');
      const result = floor(arr);
      expect(result.toArray()).toEqual([1, 1, -2, -2]);
    });

    it('handles int32 dtype', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = floor(arr);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });
  });

  describe('rint() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.4, 1.5, 1.6, 2.5], 'float32');
      const result = rint(arr);
      expect(result.toArray()).toEqual([1, 2, 2, 2]);
    });

    it('handles int32 dtype (no change)', () => {
      const arr = array([1, 2, 3], 'int32');
      const result = rint(arr);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles Infinity in rint', () => {
      const arr = array([Infinity, -Infinity]);
      const result = rint(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBe(Infinity);
      expect(values[1]).toBe(-Infinity);
    });

    it('handles NaN in rint', () => {
      const arr = array([NaN]);
      const result = rint(arr);
      const values = result.toArray() as number[];
      expect(values[0]).toBeNaN();
    });

    it('handles half-to-even for negative numbers', () => {
      const arr = array([-0.5, -1.5, -2.5, -3.5]);
      const result = rint(arr);
      expect(result.toArray()).toEqual([0, -2, -2, -4]);
    });
  });

  describe('round() additional branches', () => {
    it('handles float32 dtype', () => {
      const arr = array([1.234, 2.567], 'float32');
      const result = round(arr, 2);
      expect(result.toArray()[0]).toBeCloseTo(1.23, 2);
      expect(result.toArray()[1]).toBeCloseTo(2.57, 2);
    });
  });
});

describe('trunc() - standalone wrapper coverage', () => {
  it('truncates positive fractional values', () => {
    const arr = array([0.1, 0.9, 1.0, 1.5, 99.99]);
    const result = trunc(arr);
    expect(result.toArray()).toEqual([0, 0, 1, 1, 99]);
  });

  it('truncates negative fractional values', () => {
    const arr = array([-0.1, -0.9, -1.0, -1.5, -99.99]);
    const result = trunc(arr);
    const values = result.toArray() as number[];
    // Math.trunc(-0.1) = -0, Math.trunc(-0.9) = -0
    expect(values[2]).toBe(-1);
    expect(values[3]).toBe(-1);
    expect(values[4]).toBe(-99);
  });

  it('truncates 3D array', () => {
    const arr = array([
      [
        [1.9, -2.9],
        [3.1, -4.1],
      ],
    ]);
    const result = trunc(arr);
    expect(result.toArray()).toEqual([
      [
        [1, -2],
        [3, -4],
      ],
    ]);
  });

  it('handles zero values', () => {
    const arr = array([0, 0.0, -0.0]);
    const result = trunc(arr);
    const values = result.toArray() as number[];
    expect(values[0]).toBe(0);
  });

  it('preserves integer values', () => {
    const arr = array([1, -2, 3, -4, 0]);
    const result = trunc(arr);
    expect(result.toArray()).toEqual([1, -2, 3, -4, 0]);
  });
});

describe('around with optional parameters', () => {
  it('around without decimals parameter (default=0)', () => {
    const a = array([1.5, 2.7, 3.2]);
    const result = around(a);
    expect(result.toArray()).toEqual([2, 3, 3]);
  });

  it('around with decimals=2', () => {
    const a = array([1.567, 2.789]);
    const result = around(a, 2);
    const vals = result.toArray() as number[];
    expect(vals[0]).toBeCloseTo(1.57, 2);
    expect(vals[1]).toBeCloseTo(2.79, 2);
  });

  it('around with negative decimals', () => {
    const a = array([123, 456, 789]);
    const result = around(a, -1);
    expect(result.toArray()).toEqual([120, 460, 790]);
  });

  it('around with decimals=0 (explicit)', () => {
    const a = array([1.4, 1.5, 1.6, 2.5]);
    const result = around(a, 0);
    expect(result.toArray()).toEqual([1, 2, 2, 2]);
  });

  it('round (alias) without decimals parameter', () => {
    const a = array([1.5, 2.7, 3.2]);
    const result = round(a);
    expect(result.toArray()).toEqual([2, 3, 3]);
  });

  it('round with decimals=1', () => {
    const a = array([1.56, 2.74]);
    const result = round(a, 1);
    const vals = result.toArray() as number[];
    expect(vals[0]).toBeCloseTo(1.6, 1);
    expect(vals[1]).toBeCloseTo(2.7, 1);
  });

  it('round_ alias without decimals parameter', () => {
    const a = array([1.5, 2.7, 3.2]);
    const result = round_(a);
    expect(result.toArray()).toEqual([2, 3, 3]);
  });

  it('round_ alias with decimals=2', () => {
    const a = array([1.567, 2.789]);
    const result = round_(a, 2);
    const vals = result.toArray() as number[];
    expect(vals[0]).toBeCloseTo(1.57, 2);
    expect(vals[1]).toBeCloseTo(2.79, 2);
  });
});

describe('Non-contiguous array tests', () => {
  it('around with non-contiguous array', () => {
    const a = array([
      [1.4, 1.5],
      [1.6, 2.5],
    ]);
    const result = around(a.T);
    expect(result.shape).toEqual([2, 2]);
  });

  it('ceil with non-contiguous array', () => {
    const a = array([
      [1.1, 1.5],
      [1.9, 2.3],
    ]);
    const result = ceil(a.T);
    expect(result.shape).toEqual([2, 2]);
  });

  it('floor with non-contiguous array', () => {
    const a = array([
      [1.1, 1.5],
      [1.9, 2.3],
    ]);
    const result = floor(a.T);
    expect(result.shape).toEqual([2, 2]);
  });

  it('rint with non-contiguous array', () => {
    const a = array([
      [1.4, 1.6],
      [2.3, 2.7],
    ]);
    const result = rint(a.T);
    expect(result.shape).toEqual([2, 2]);
  });

  it('trunc with non-contiguous array', () => {
    const a = array([
      [1.7, -1.7],
      [2.3, -2.3],
    ]);
    const result = trunc(a.T);
    expect(result.shape).toEqual([2, 2]);
  });
});
