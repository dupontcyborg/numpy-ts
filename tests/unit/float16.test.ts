/**
 * Float16 dtype support tests
 */

import { describe, expect, it } from 'vitest';
import { arange, array, empty, full, linspace, ones, zeros } from '../../src';
import {
  getDTypeSize,
  hasFloat16,
  isFloatDType,
  isValidDType,
  promoteDTypes,
} from '../../src/common/dtype';
import { float16BitsToNumber, numberToFloat16Bits } from '../../src/common/float16-conv';

describe('Float16 Support', () => {
  describe('dtype system', () => {
    it('float16 is a valid dtype', () => {
      expect(isValidDType('float16')).toBe(true);
    });

    it('float16 is a float dtype', () => {
      expect(isFloatDType('float16')).toBe(true);
    });

    it('float16 has size 2', () => {
      expect(getDTypeSize('float16')).toBe(2);
    });

    it('exports hasFloat16 runtime flag', () => {
      expect(typeof hasFloat16).toBe('boolean');
    });
  });

  describe('array creation', () => {
    it('creates zeros with float16 dtype', () => {
      const a = zeros([3], 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.shape).toEqual([3]);
      expect(a.toArray()).toEqual([0, 0, 0]);
    });

    it('creates ones with float16 dtype', () => {
      const a = ones([2, 2], 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    it('creates array from data with float16 dtype', () => {
      const a = array([1, 2, 3], 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.toArray()).toEqual([1, 2, 3]);
    });

    it('creates empty with float16 dtype', () => {
      const a = empty([4], 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.shape).toEqual([4]);
    });

    it('creates full with float16 dtype', () => {
      const a = full([3], 42, 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.toArray()).toEqual([42, 42, 42]);
    });

    it('creates arange with float16 dtype', () => {
      const a = arange(0, 5, 1, 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.toArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('creates linspace with float16 dtype', () => {
      const a = linspace(0, 1, 3, 'float16');
      expect(a.dtype).toBe('float16');
      expect(a.shape).toEqual([3]);
    });

    it('creates 2D array with float16 dtype', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'float16',
      );
      expect(a.dtype).toBe('float16');
      expect(a.shape).toEqual([2, 2]);
      expect(a.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('arithmetic', () => {
    it('add two float16 arrays', () => {
      const a = array([1, 2, 3], 'float16');
      const b = array([4, 5, 6], 'float16');
      const c = a.add(b);
      expect(c.dtype).toBe('float16');
      expect(c.toArray()).toEqual([5, 7, 9]);
    });

    it('multiply float16 arrays', () => {
      const a = array([2, 3, 4], 'float16');
      const b = array([3, 4, 5], 'float16');
      const c = a.multiply(b);
      expect(c.dtype).toBe('float16');
      expect(c.toArray()).toEqual([6, 12, 20]);
    });

    it('subtract float16 arrays', () => {
      const a = array([10, 20, 30], 'float16');
      const b = array([1, 2, 3], 'float16');
      const c = a.subtract(b);
      expect(c.dtype).toBe('float16');
      expect(c.toArray()).toEqual([9, 18, 27]);
    });
  });

  describe('type promotion', () => {
    it('float16 + float16 = float16', () => {
      expect(promoteDTypes('float16', 'float16')).toBe('float16');
    });

    it('float16 + float32 = float32', () => {
      expect(promoteDTypes('float16', 'float32')).toBe('float32');
      expect(promoteDTypes('float32', 'float16')).toBe('float32');
    });

    it('float16 + float64 = float64', () => {
      expect(promoteDTypes('float16', 'float64')).toBe('float64');
      expect(promoteDTypes('float64', 'float16')).toBe('float64');
    });

    it('float16 + int8/uint8 = float16', () => {
      expect(promoteDTypes('float16', 'int8')).toBe('float16');
      expect(promoteDTypes('float16', 'uint8')).toBe('float16');
      expect(promoteDTypes('int8', 'float16')).toBe('float16');
      expect(promoteDTypes('uint8', 'float16')).toBe('float16');
    });

    it('float16 + int16/uint16 = float32', () => {
      expect(promoteDTypes('float16', 'int16')).toBe('float32');
      expect(promoteDTypes('float16', 'uint16')).toBe('float32');
      expect(promoteDTypes('int16', 'float16')).toBe('float32');
      expect(promoteDTypes('uint16', 'float16')).toBe('float32');
    });

    it('float16 + int32/uint32/int64/uint64 = float64', () => {
      expect(promoteDTypes('float16', 'int32')).toBe('float64');
      expect(promoteDTypes('float16', 'uint32')).toBe('float64');
      expect(promoteDTypes('float16', 'int64')).toBe('float64');
      expect(promoteDTypes('float16', 'uint64')).toBe('float64');
    });

    it('bool + float16 = float16', () => {
      expect(promoteDTypes('bool', 'float16')).toBe('float16');
      expect(promoteDTypes('float16', 'bool')).toBe('float16');
    });

    it('array arithmetic promotes correctly', () => {
      const f16 = array([1, 2], 'float16');
      const f32 = array([1, 2], 'float32');
      const f64 = array([1, 2], 'float64');

      expect(f16.add(f32).dtype).toBe('float32');
      expect(f16.add(f64).dtype).toBe('float64');
      expect(f16.add(f16).dtype).toBe('float16');
    });
  });

  describe('astype', () => {
    it('float16 to float32', () => {
      const a = array([1.5, 2.5, 3.5], 'float16');
      const b = a.astype('float32');
      expect(b.dtype).toBe('float32');
      expect(b.toArray()).toEqual([1.5, 2.5, 3.5]);
    });

    it('float16 to float64', () => {
      const a = array([1.5, 2.5], 'float16');
      const b = a.astype('float64');
      expect(b.dtype).toBe('float64');
      expect(b.toArray()).toEqual([1.5, 2.5]);
    });

    it('float32 to float16', () => {
      const a = array([1.5, 2.5, 3.5], 'float32');
      const b = a.astype('float16');
      expect(b.dtype).toBe('float16');
      expect(b.toArray()).toEqual([1.5, 2.5, 3.5]);
    });

    it('int32 to float16', () => {
      const a = array([1, 2, 3], 'int32');
      const b = a.astype('float16');
      expect(b.dtype).toBe('float16');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('float16 to int32', () => {
      const a = array([1.5, 2.7, 3.1], 'float16');
      const b = a.astype('int32');
      expect(b.dtype).toBe('int32');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });
  });

  describe('operations', () => {
    it('sum of float16 array', () => {
      const a = array([1, 2, 3, 4], 'float16');
      const s = a.sum();
      // sum may return number or NDArray depending on implementation
      const val = typeof s === 'number' ? s : (s as any).item();
      expect(val).toBe(10);
    });

    it('mean of float16 array', () => {
      const a = array([2, 4, 6, 8], 'float16');
      const m = a.mean();
      const val = typeof m === 'number' ? m : (m as any).item();
      expect(val).toBe(5);
    });

    it('reshape float16 array', () => {
      const a = array([1, 2, 3, 4, 5, 6], 'float16');
      const b = a.reshape([2, 3]);
      expect(b.dtype).toBe('float16');
      expect(b.shape).toEqual([2, 3]);
      expect(b.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('transpose float16 array', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'float16',
      );
      const b = a.T;
      expect(b.dtype).toBe('float16');
      expect(b.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it('copy float16 array', () => {
      const a = array([1, 2, 3], 'float16');
      const b = a.copy();
      expect(b.dtype).toBe('float16');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('slice float16 array', () => {
      const a = array([1, 2, 3, 4, 5], 'float16');
      const b = a.slice('1:4');
      expect(b.dtype).toBe('float16');
      expect(b.toArray()).toEqual([2, 3, 4]);
    });
  });

  describe('float16 bit conversion', () => {
    it('converts zero', () => {
      expect(float16BitsToNumber(0x0000)).toBe(0);
      expect(numberToFloat16Bits(0)).toBe(0);
    });

    it('converts one', () => {
      expect(float16BitsToNumber(0x3c00)).toBe(1);
      expect(numberToFloat16Bits(1)).toBe(0x3c00);
    });

    it('converts negative one', () => {
      expect(float16BitsToNumber(0xbc00)).toBe(-1);
      expect(numberToFloat16Bits(-1)).toBe(0xbc00);
    });

    it('converts infinity', () => {
      expect(float16BitsToNumber(0x7c00)).toBe(Infinity);
      expect(float16BitsToNumber(0xfc00)).toBe(-Infinity);
      expect(numberToFloat16Bits(Infinity)).toBe(0x7c00);
      expect(numberToFloat16Bits(-Infinity)).toBe(0xfc00);
    });

    it('converts NaN', () => {
      expect(float16BitsToNumber(0x7e00)).toBeNaN();
      expect(isNaN(float16BitsToNumber(numberToFloat16Bits(NaN)))).toBe(true);
    });

    it('converts 0.5', () => {
      expect(float16BitsToNumber(0x3800)).toBe(0.5);
      expect(numberToFloat16Bits(0.5)).toBe(0x3800);
    });

    it('round-trips common values', () => {
      const values = [0, 1, -1, 0.5, -0.5, 2, 10, 100, 0.1, 0.25];
      for (const v of values) {
        const bits = numberToFloat16Bits(v);
        const back = float16BitsToNumber(bits);
        // float16 has limited precision, so check approximate equality
        if (v === 0) {
          expect(back).toBe(0);
        } else {
          expect(Math.abs(back - v) / Math.abs(v)).toBeLessThan(0.01);
        }
      }
    });
  });
});
