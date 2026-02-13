import { describe, it, expect } from 'vitest';
import {
  array,
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  arctan2,
  hypot,
  degrees,
  radians,
  deg2rad,
  rad2deg,
} from '../../src';

describe('Trigonometric Operations', () => {
  describe('sin', () => {
    it('computes sine of angles', () => {
      const arr = array([0, Math.PI / 6, Math.PI / 4, Math.PI / 2]);
      const result = sin(arr);
      const expected = [0, 0.5, Math.SQRT1_2, 1];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, Math.PI / 2, Math.PI]);
      const result = arr.sin();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[2]!)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = sin(arr);
      expect(result.dtype).toBe('float64');
    });

    it('preserves float32 dtype', () => {
      const arr = array([0, 1], 'float32');
      const result = sin(arr);
      expect(result.dtype).toBe('float32');
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [0, Math.PI / 2],
        [Math.PI, (3 * Math.PI) / 2],
      ]);
      const result = sin(arr);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('cos', () => {
    it('computes cosine of angles', () => {
      const arr = array([0, Math.PI / 3, Math.PI / 2, Math.PI]);
      const result = cos(arr);
      const expected = [1, 0.5, 0, -1];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, Math.PI]);
      const result = arr.cos();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - -1)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = cos(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('tan', () => {
    it('computes tangent of angles', () => {
      const arr = array([0, Math.PI / 4]);
      const result = tan(arr);
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - 1)).toBeLessThan(1e-10);
    });

    it('works as method', () => {
      const arr = array([0, Math.PI / 4]);
      const result = arr.tan();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - 1)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1], 'int32');
      const result = tan(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('arcsin', () => {
    it('computes inverse sine', () => {
      const arr = array([0, 0.5, 1]);
      const result = arcsin(arr);
      const expected = [0, Math.PI / 6, Math.PI / 2];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.arcsin();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.PI / 2)).toBeLessThan(1e-10);
    });

    it('returns NaN for out-of-range values', () => {
      const arr = array([2]);
      const result = arcsin(arr);
      expect(Number.isNaN(result.get([0]))).toBe(true);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1], 'int32');
      const result = arcsin(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('arccos', () => {
    it('computes inverse cosine', () => {
      const arr = array([1, 0.5, 0]);
      const result = arccos(arr);
      const expected = [0, Math.PI / 3, Math.PI / 2];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([1, 0]);
      const result = arr.arccos();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.PI / 2)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1], 'int32');
      const result = arccos(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('arctan', () => {
    it('computes inverse tangent', () => {
      const arr = array([0, 1]);
      const result = arctan(arr);
      const expected = [0, Math.PI / 4];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([0, 1]);
      const result = arr.arctan();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]!)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - Math.PI / 4)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = arctan(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('arctan2', () => {
    it('computes four-quadrant inverse tangent', () => {
      const y = array([1, 1, -1, -1]);
      const x = array([1, -1, 1, -1]);
      const result = arctan2(y, x);
      const expected = [Math.PI / 4, (3 * Math.PI) / 4, -Math.PI / 4, (-3 * Math.PI) / 4];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works with scalar x', () => {
      const y = array([1, 2, 3]);
      const result = arctan2(y, 1);
      const expected = [Math.PI / 4, Math.atan(2), Math.atan(3)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const y = array([1, 0]);
      const x = array([0, 1]);
      const result = y.arctan2(x);
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]! - Math.PI / 2)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]!)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const y = array([1, 2], 'int32');
      const x = array([1, 2], 'int32');
      const result = arctan2(y, x);
      expect(result.dtype).toBe('float64');
    });

    it('handles BigInt arrays with scalar', () => {
      const y = array([1n, 2n, 3n], 'int64');
      const result = arctan2(y, 1);
      expect(result.dtype).toBe('float64');
      const expected = [Math.PI / 4, Math.atan(2), Math.atan(3)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });
  });

  describe('hypot', () => {
    it('computes hypotenuse', () => {
      const x1 = array([3, 5, 8]);
      const x2 = array([4, 12, 15]);
      const result = hypot(x1, x2);
      const expected = [5, 13, 17];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works with scalar', () => {
      const x1 = array([3, 6]);
      const result = hypot(x1, 4);
      const expected = [5, Math.sqrt(52)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([3, 5]);
      const other = array([4, 12]);
      const result = arr.hypot(other);
      const expected = [5, 13];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('promotes integer arrays to float64', () => {
      const x1 = array([3, 4], 'int32');
      const x2 = array([4, 3], 'int32');
      const result = hypot(x1, x2);
      expect(result.dtype).toBe('float64');
    });

    it('handles BigInt arrays with scalar', () => {
      const x1 = array([3n, 6n], 'int64');
      const result = hypot(x1, 4);
      expect(result.dtype).toBe('float64');
      const expected = [5, Math.sqrt(52)];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });
  });

  describe('degrees', () => {
    it('converts radians to degrees', () => {
      const arr = array([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
      const result = degrees(arr);
      const expected = [0, 30, 45, 90, 180];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([Math.PI, 2 * Math.PI]);
      const result = arr.degrees();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]! - 180)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - 360)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = degrees(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('radians', () => {
    it('converts degrees to radians', () => {
      const arr = array([0, 30, 45, 90, 180]);
      const result = radians(arr);
      const expected = [0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('works as method', () => {
      const arr = array([180, 360]);
      const result = arr.radians();
      const resultArr = result.toArray() as number[];
      expect(Math.abs(resultArr[0]! - Math.PI)).toBeLessThan(1e-10);
      expect(Math.abs(resultArr[1]! - 2 * Math.PI)).toBeLessThan(1e-10);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 90, 180], 'int32');
      const result = radians(arr);
      expect(result.dtype).toBe('float64');
    });

    it('roundtrip with degrees', () => {
      const original = array([0, 30, 45, 90, 180, 270, 360]);
      const asRadians = radians(original);
      const backToDegrees = degrees(asRadians);
      const resultArr = backToDegrees.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });
  });

  describe('deg2rad', () => {
    it('converts degrees to radians (alias for radians)', () => {
      const arr = array([0, 30, 45, 90, 180]);
      const result = deg2rad(arr);
      const expected = [0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('produces same result as radians', () => {
      const arr = array([0, 45, 90, 180]);
      const result1 = deg2rad(arr);
      const result2 = radians(arr);
      const arr1 = result1.toArray() as number[];
      const arr2 = result2.toArray() as number[];
      for (let i = 0; i < arr1.length; i++) {
        expect(arr1[i]).toBe(arr2[i]);
      }
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 90, 180], 'int32');
      const result = deg2rad(arr);
      expect(result.dtype).toBe('float64');
    });
  });

  describe('rad2deg', () => {
    it('converts radians to degrees (alias for degrees)', () => {
      const arr = array([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
      const result = rad2deg(arr);
      const expected = [0, 30, 45, 90, 180];
      const resultArr = result.toArray() as number[];
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(resultArr[i]! - expected[i]!)).toBeLessThan(1e-10);
      }
    });

    it('produces same result as degrees', () => {
      const arr = array([0, Math.PI / 4, Math.PI / 2, Math.PI]);
      const result1 = rad2deg(arr);
      const result2 = degrees(arr);
      const arr1 = result1.toArray() as number[];
      const arr2 = result2.toArray() as number[];
      for (let i = 0; i < arr1.length; i++) {
        expect(arr1[i]).toBe(arr2[i]);
      }
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([0, 1, 2], 'int32');
      const result = rad2deg(arr);
      expect(result.dtype).toBe('float64');
    });

    it('roundtrip with deg2rad', () => {
      const original = array([0, 30, 45, 90, 180, 270, 360]);
      const asRadians = deg2rad(original);
      const backToDegrees = rad2deg(asRadians);
      const resultArr = backToDegrees.toArray() as number[];
      const originalArr = original.toArray() as number[];
      for (let i = 0; i < resultArr.length; i++) {
        expect(Math.abs(resultArr[i]! - originalArr[i]!)).toBeLessThan(1e-10);
      }
    });
  });
});
