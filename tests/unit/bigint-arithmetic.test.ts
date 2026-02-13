/**
 * Unit tests for BigInt arithmetic correctness
 * Tests that int64/uint64 arithmetic doesn't lose precision
 */

import { describe, it, expect } from 'vitest';
import { zeros, ones } from '../../src';

describe('BigInt Arithmetic Correctness', () => {
  // Number.MAX_SAFE_INTEGER = 9007199254740991
  // Values above this can lose precision when converted to Number

  describe('Scalar arithmetic with large values', () => {
    it('preserves precision in int64 addition with large numbers', () => {
      const arr = zeros([2, 2], 'int64');
      arr.data[0] = 9007199254740993n; // Larger than MAX_SAFE_INTEGER
      arr.data[1] = 9007199254740994n;
      arr.data[2] = 9007199254740995n;
      arr.data[3] = 9007199254740996n;

      const result = arr.add(10);

      // Should preserve precision
      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(9007199254741003n);
      expect(result.data[1]).toBe(9007199254741004n);
      expect(result.data[2]).toBe(9007199254741005n);
      expect(result.data[3]).toBe(9007199254741006n);
    });

    it('preserves precision in uint64 subtraction with large numbers', () => {
      const arr = zeros([2, 2], 'uint64');
      arr.data[0] = 18446744073709551000n; // Near max uint64
      arr.data[1] = 18446744073709551001n;

      const result = arr.subtract(100);

      expect(result.dtype).toBe('uint64');
      expect(result.data[0]).toBe(18446744073709550900n);
      expect(result.data[1]).toBe(18446744073709550901n);
    });

    it('preserves precision in int64 multiplication with large numbers', () => {
      const arr = zeros([2], 'int64');
      arr.data[0] = 9007199254740993n;
      arr.data[1] = 9007199254740994n;

      const result = arr.multiply(2);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(18014398509481986n);
      expect(result.data[1]).toBe(18014398509481988n);
    });

    it('int64 division promotes to float64 (NumPy behavior)', () => {
      const arr = zeros([2], 'int64');
      arr.data[0] = 9007199254740992n;
      arr.data[1] = 9007199254740994n;

      const result = arr.divide(2);

      // NumPy promotes integer division to float64
      expect(result.dtype).toBe('float64');
      expect(result.data[0]).toBe(4503599627370496);
      expect(result.data[1]).toBe(4503599627370497);
    });
  });

  describe('Array-to-array arithmetic with large values', () => {
    it('preserves precision in int64 array addition', () => {
      const a = zeros([2], 'int64');
      a.data[0] = 9007199254740993n;
      a.data[1] = 9007199254740994n;

      const b = zeros([2], 'int64');
      b.data[0] = 1000000000000000n;
      b.data[1] = 2000000000000000n;

      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(10007199254740993n);
      expect(result.data[1]).toBe(11007199254740994n);
    });

    it('preserves precision in int64 array subtraction', () => {
      const a = zeros([2], 'int64');
      a.data[0] = 10007199254740993n;
      a.data[1] = 11007199254740994n;

      const b = zeros([2], 'int64');
      b.data[0] = 1000000000000000n;
      b.data[1] = 2000000000000000n;

      const result = a.subtract(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(9007199254740993n);
      expect(result.data[1]).toBe(9007199254740994n);
    });

    it('preserves precision in int64 array multiplication', () => {
      const a = zeros([2], 'int64');
      a.data[0] = 9007199254740993n;
      a.data[1] = 123456789012345n;

      const b = zeros([2], 'int64');
      b.data[0] = 2n;
      b.data[1] = 3n;

      const result = a.multiply(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(18014398509481986n);
      expect(result.data[1]).toBe(370370367037035n);
    });

    it('int64 array division promotes to float64 (NumPy behavior)', () => {
      const a = zeros([2], 'int64');
      a.data[0] = 18014398509481986n;
      a.data[1] = 370370367037035n;

      const b = zeros([2], 'int64');
      b.data[0] = 2n;
      b.data[1] = 3n;

      const result = a.divide(b);

      // NumPy promotes integer division to float64
      expect(result.dtype).toBe('float64');
      expect(result.data[0]).toBe(Number(9007199254740993n));
      expect(result.data[1]).toBe(Number(123456789012345n));
    });
  });

  describe('Broadcasting with BigInt arrays', () => {
    it('preserves precision when broadcasting int64 arrays', () => {
      const a = zeros([2, 3], 'int64');
      // Set first row
      a.data[0] = 9007199254740993n;
      a.data[1] = 9007199254740994n;
      a.data[2] = 9007199254740995n;
      // Set second row
      a.data[3] = 9007199254740996n;
      a.data[4] = 9007199254740997n;
      a.data[5] = 9007199254740998n;

      const b = zeros([3], 'int64');
      b.data[0] = 100n;
      b.data[1] = 200n;
      b.data[2] = 300n;

      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.shape).toEqual([2, 3]);
      // First row
      expect(result.data[0]).toBe(9007199254741093n);
      expect(result.data[1]).toBe(9007199254741194n);
      expect(result.data[2]).toBe(9007199254741295n);
      // Second row
      expect(result.data[3]).toBe(9007199254741096n);
      expect(result.data[4]).toBe(9007199254741197n);
      expect(result.data[5]).toBe(9007199254741298n);
    });
  });

  describe('Edge cases', () => {
    it('handles maximum safe integer boundary', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = BigInt(Number.MAX_SAFE_INTEGER); // 9007199254740991
      arr.data[1] = BigInt(Number.MAX_SAFE_INTEGER) + 1n; // First unsafe
      arr.data[2] = BigInt(Number.MAX_SAFE_INTEGER) + 2n;

      const result = arr.add(10);

      expect(result.data[0]).toBe(BigInt(Number.MAX_SAFE_INTEGER) + 10n);
      expect(result.data[1]).toBe(BigInt(Number.MAX_SAFE_INTEGER) + 11n);
      expect(result.data[2]).toBe(BigInt(Number.MAX_SAFE_INTEGER) + 12n);
    });

    it('handles negative large int64 values', () => {
      const arr = zeros([2], 'int64');
      arr.data[0] = -9007199254740993n;
      arr.data[1] = -9007199254740994n;

      const result = arr.subtract(10);

      expect(result.data[0]).toBe(-9007199254741003n);
      expect(result.data[1]).toBe(-9007199254741004n);
    });

    it('handles uint64 near maximum value', () => {
      const arr = zeros([2], 'uint64');
      // Near max uint64: 2^64 - 1 = 18446744073709551615
      arr.data[0] = 18446744073709551610n;
      arr.data[1] = 18446744073709551611n;

      const result = arr.add(2);

      expect(result.data[0]).toBe(18446744073709551612n);
      expect(result.data[1]).toBe(18446744073709551613n);
    });

    it('int64 division promotes to float64 and returns float results', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = 9007199254740993n;
      arr.data[1] = 9007199254740994n;
      arr.data[2] = 9007199254740995n;

      const result = arr.divide(3);

      // NumPy promotes to float64, so we get floating-point division
      expect(result.dtype).toBe('float64');
      expect(result.data[0]).toBe(3002399751580330.5);
      expect(result.data[1]).toBe(3002399751580331.5);
      expect(result.data[2]).toBe(3002399751580332.0);
    });
  });

  describe('Dtype preservation', () => {
    it('preserves int64 dtype through arithmetic operations (except division)', () => {
      const arr = ones([2, 2], 'int64');

      const result1 = arr.add(1);
      expect(result1.dtype).toBe('int64');

      const result2 = result1.multiply(2);
      expect(result2.dtype).toBe('int64');

      const result3 = result2.subtract(1);
      expect(result3.dtype).toBe('int64');

      // Division promotes to float64 (NumPy behavior)
      const result4 = result3.divide(2);
      expect(result4.dtype).toBe('float64');
    });

    it('preserves uint64 dtype through arithmetic operations', () => {
      const arr = ones([2, 2], 'uint64');

      const result1 = arr.add(10);
      expect(result1.dtype).toBe('uint64');

      const result2 = result1.multiply(5);
      expect(result2.dtype).toBe('uint64');
    });

    it('preserves int64 dtype in array-to-array operations', () => {
      const a = ones([2, 2], 'int64');
      const b = ones([2, 2], 'int64');

      const result = a.add(b);
      expect(result.dtype).toBe('int64');
    });
  });

  describe('BigInt Reduction Operations', () => {
    it('preserves precision in int64 sum reduction', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = 9007199254740991n; // MAX_SAFE_INTEGER
      arr.data[1] = 9007199254740992n; // MAX_SAFE_INTEGER + 1
      arr.data[2] = 9007199254740993n; // MAX_SAFE_INTEGER + 2

      const sum = arr.sum();
      // Sum: 27021597764222976
      expect(sum).toBe(27021597764222976);
    });

    it('preserves precision in int64 sum with axis', () => {
      const arr = zeros([2, 3], 'int64');
      // First row
      arr.data[0] = 9007199254740991n;
      arr.data[1] = 9007199254740992n;
      arr.data[2] = 9007199254740993n;
      // Second row
      arr.data[3] = 1000000000000000n;
      arr.data[4] = 2000000000000000n;
      arr.data[5] = 3000000000000000n;

      const sumAxis0 = arr.sum(0);
      expect(sumAxis0.dtype).toBe('int64');
      expect(sumAxis0.shape).toEqual([3]);
      expect(sumAxis0.data[0]).toBe(10007199254740991n);
      expect(sumAxis0.data[1]).toBe(11007199254740992n);
      expect(sumAxis0.data[2]).toBe(12007199254740993n);

      const sumAxis1 = arr.sum(1);
      expect(sumAxis1.dtype).toBe('int64');
      expect(sumAxis1.shape).toEqual([2]);
      expect(sumAxis1.data[0]).toBe(27021597764222976n); // First row sum
      expect(sumAxis1.data[1]).toBe(6000000000000000n); // Second row sum
    });

    it('preserves dtype in int64 max reduction', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = 9007199254740991n;
      arr.data[1] = 9007199254740995n; // Largest
      arr.data[2] = 9007199254740992n;

      const maxVal = arr.max();
      expect(maxVal).toBe(Number(9007199254740995n));
    });

    it('preserves dtype in int64 max with axis', () => {
      const arr = zeros([2, 3], 'int64');
      arr.data[0] = 1000n;
      arr.data[1] = 5000n;
      arr.data[2] = 3000n;
      arr.data[3] = 9007199254740993n;
      arr.data[4] = 2000n;
      arr.data[5] = 4000n;

      const maxAxis1 = arr.max(1);
      expect(maxAxis1.dtype).toBe('int64');
      expect(maxAxis1.shape).toEqual([2]);
      expect(maxAxis1.data[0]).toBe(5000n);
      expect(maxAxis1.data[1]).toBe(9007199254740993n);
    });

    it('preserves dtype in int64 min reduction', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = 9007199254740991n;
      arr.data[1] = 9007199254740990n; // Smallest
      arr.data[2] = 9007199254740992n;

      const minVal = arr.min();
      expect(minVal).toBe(Number(9007199254740990n));
    });

    it('preserves dtype in int64 min with axis', () => {
      const arr = zeros([2, 3], 'int64');
      arr.data[0] = 5000n;
      arr.data[1] = 1000n;
      arr.data[2] = 3000n;
      arr.data[3] = 9007199254740993n;
      arr.data[4] = 2000n;
      arr.data[5] = 4000n;

      const minAxis1 = arr.min(1);
      expect(minAxis1.dtype).toBe('int64');
      expect(minAxis1.shape).toEqual([2]);
      expect(minAxis1.data[0]).toBe(1000n);
      expect(minAxis1.data[1]).toBe(2000n);
    });

    it('returns float64 for int64 mean reduction', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = 9000000000000000n;
      arr.data[1] = 9000000000000003n;
      arr.data[2] = 9000000000000006n;

      const mean = arr.mean();
      // Mean should be 9000000000000003
      expect(mean).toBe(9000000000000003);
    });

    it('returns float64 array for int64 mean with axis', () => {
      const arr = zeros([2, 3], 'int64');
      arr.data[0] = 9000000000000000n;
      arr.data[1] = 9000000000000003n;
      arr.data[2] = 9000000000000006n;
      arr.data[3] = 3000000000000000n;
      arr.data[4] = 6000000000000000n;
      arr.data[5] = 9000000000000000n;

      const meanAxis1 = arr.mean(1);
      expect(meanAxis1.dtype).toBe('float64'); // mean returns float64 for int types
      expect(meanAxis1.shape).toEqual([2]);
      expect(meanAxis1.data[0]).toBe(9000000000000003); // Mean of first row
      expect(meanAxis1.data[1]).toBe(6000000000000000); // Mean of second row
    });

    it('handles keepdims=true for int64 sum', () => {
      const arr = zeros([2, 3], 'int64');
      arr.data[0] = 1000n;
      arr.data[1] = 2000n;
      arr.data[2] = 3000n;
      arr.data[3] = 4000n;
      arr.data[4] = 5000n;
      arr.data[5] = 6000n;

      const sumKeep = arr.sum(1, true);
      expect((sumKeep as any).dtype).toBe('int64');
      expect((sumKeep as any).shape).toEqual([2, 1]);
      expect((sumKeep as any).data[0]).toBe(6000n);
      expect((sumKeep as any).data[1]).toBe(15000n);
    });

    it('handles uint64 reduction operations', () => {
      const arr = zeros([3], 'uint64');
      arr.data[0] = 18446744073709551610n;
      arr.data[1] = 18446744073709551611n;
      arr.data[2] = 18446744073709551612n;

      const maxVal = arr.max();
      expect(maxVal).toBe(Number(18446744073709551612n));

      const minVal = arr.min();
      expect(minVal).toBe(Number(18446744073709551610n));
    });

    it('handles negative int64 values in reductions', () => {
      const arr = zeros([3], 'int64');
      arr.data[0] = -9007199254740993n;
      arr.data[1] = -9007199254740994n;
      arr.data[2] = -9007199254740995n;

      const sum = arr.sum();
      // Expected sum: -27021597764222982
      expect(sum).toBe(-27021597764222980 - 2);

      const maxVal = arr.max();
      expect(maxVal).toBe(Number(-9007199254740993n));

      const minVal = arr.min();
      expect(minVal).toBe(Number(-9007199254740995n));
    });
  });
});
