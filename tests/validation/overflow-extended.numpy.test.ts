/**
 * Extended overflow validation tests
 * Cross-validates reduction overflow, math function overflow, casting overflow,
 * matmul accumulation overflow, and bitwise overflow against NumPy.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../src/full/index';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

/** Extract numeric value from a result that may be NDArray or scalar */
function toNum(x: unknown): number {
  if (typeof x === 'number') return x;
  if (typeof x === 'bigint') return Number(x);
  if (x && typeof x === 'object' && 'tolist' in x) {
    const v = (x as { tolist(): unknown }).tolist();
    return typeof v === 'number' ? v : Number(v);
  }
  return Number(x);
}

describe('NumPy Validation: Extended Overflow Behavior', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  // ============================================================
  // Reduction Overflow
  // ============================================================
  describe('reduction overflow', () => {
    it('int8 sum promotes to avoid overflow', () => {
      const r = np.sum(np.array([127, 1], 'int8'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.sum(np.array([127, 1], dtype=np.int8)))
      `);

      expect(toNum(r)).toBe(npResult.value);
    });

    it('int8 sum of large arange promotes correctly', () => {
      const r = np.sum(np.arange(0, 200, 1, 'int8'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.sum(np.arange(200, dtype=np.int8)))
      `);

      expect(toNum(r)).toBe(npResult.value);
    });

    it('uint8 prod promotes to avoid overflow', () => {
      const r = np.prod(np.array([16, 16, 16], 'uint8'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.prod(np.array([16, 16, 16], dtype=np.uint8)))
      `);

      expect(toNum(r)).toBe(npResult.value);
    });

    it('int16 cumsum promotes to avoid overflow', () => {
      const r = np.cumsum(np.array([32000, 1000, 1000], 'int16'));

      const npResult = runNumPy(`
import numpy as np
result = np.cumsum(np.array([32000, 1000, 1000], dtype=np.int16)).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('int32 sum promotes to avoid overflow', () => {
      const r = np.sum(np.array([2147483647, 1], 'int32'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.sum(np.array([2147483647, 1], dtype=np.int32)))
      `);

      expect(toNum(r)).toBe(npResult.value);
    });

    it('float32 sum with large values', () => {
      const r = np.sum(np.full([1000], 1e35, 'float32'));

      const npResult = runNumPy(`
import numpy as np
result = float(np.sum(np.full(1000, 1e35, dtype=np.float32)))
      `);

      const relErr = Math.abs(toNum(r) - npResult.value) / Math.abs(npResult.value);
      expect(relErr).toBeLessThan(1e-3);
    });

    it('float32 prod overflow to infinity', () => {
      const r = np.prod(np.full([100], 100.0, 'float32'));

      const npResult = runNumPy(`
import numpy as np
result = float(np.prod(np.full(100, 100.0, dtype=np.float32)))
      `);

      expect(toNum(r)).toBe(npResult.value);
      expect(toNum(r)).toBe(Infinity);
    });

    it('NaN propagates through sum and prod', () => {
      const a = np.array([NaN, 1, 2], 'float64');
      expect(toNum(np.sum(a))).toBeNaN();
      expect(toNum(np.prod(a))).toBeNaN();
    });
  });

  // ============================================================
  // Math Function Overflow
  // ============================================================
  describe('math function overflow', () => {
    it('exp(1000) overflows to inf in float64', () => {
      const r = np.exp(np.array([1000], 'float64'));

      const npResult = runNumPy(`
import numpy as np
result = float(np.exp(np.float64(1000)))
      `);

      expect(r.tolist()[0]).toBe(npResult.value);
      expect(r.tolist()[0]).toBe(Infinity);
    });

    it('exp(100) overflows to inf in float32', () => {
      const r = np.exp(np.array([100], 'float32'));

      const npResult = runNumPy(`
import numpy as np
result = float(np.exp(np.float32(100)))
      `);

      expect(r.tolist()[0]).toBe(npResult.value);
    });

    it('exp(89) overflows to inf in float32', () => {
      const r = np.exp(np.array([89], 'float32'));

      const npResult = runNumPy(`
import numpy as np
result = float(np.exp(np.float32(89)))
      `);

      expect(r.tolist()[0]).toBe(npResult.value);
    });

    it('log(0) returns -inf', () => {
      expect(np.log(np.array([0], 'float64')).tolist()[0]).toBe(-Infinity);
    });

    it('log(-1) returns NaN', () => {
      expect(np.log(np.array([-1], 'float64')).tolist()[0]).toBeNaN();
    });

    it('power(2, 1024) overflows to inf in float64', () => {
      expect(np.power(np.array([2], 'float64'), np.array([1024], 'float64')).tolist()[0]).toBe(
        Infinity
      );
    });

    it('power(2, 8) wraps to 0 in int8', () => {
      const r = np.power(np.array([2], 'int8'), np.array([8], 'int8'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.power(np.int8(2), np.int8(8)))
      `);

      expect(r.tolist()[0]).toBe(npResult.value);
      expect(r.dtype).toBe('int8');
    });

    it('sqrt(1e308) returns 1e154', () => {
      expect(np.sqrt(np.array([1e308], 'float64')).tolist()[0]).toBeCloseTo(1e154, -150);
    });

    it('sqrt(-1) returns NaN', () => {
      expect(np.sqrt(np.array([-1], 'float64')).tolist()[0]).toBeNaN();
    });
  });

  // ============================================================
  // Casting Overflow
  // ============================================================
  describe('casting overflow', () => {
    it('int32 to uint8 wraps correctly', () => {
      const r = np.array([256, -1, 300, 0], 'int32').astype('uint8');

      const npResult = runNumPy(`
import numpy as np
result = np.array([256, -1, 300, 0], dtype=np.int32).astype(np.uint8).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('int64 to int8 truncates correctly', () => {
      const r = np.array([1000, -1000], 'int64').astype('int8');

      const npResult = runNumPy(`
import numpy as np
result = np.array([1000, -1000], dtype=np.int64).astype(np.int8).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 with normal out-of-range values to int8', () => {
      const r = np.array([200, -200, 127.5, -128.5], 'float64').astype('int8');

      const npResult = runNumPy(`
import numpy as np
result = np.array([200.0, -200.0, 127.5, -128.5], dtype=np.float64).astype(np.int8).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 inf/nan to int32 matches NumPy', () => {
      const r = np.array([Infinity, -Infinity, NaN], 'float64').astype('int32');

      const npResult = runNumPy(`
import numpy as np
result = np.array([np.inf, -np.inf, np.nan], dtype=np.float64).astype(np.int32).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 inf/nan to int8 matches NumPy', () => {
      const r = np.array([Infinity, -Infinity, NaN], 'float64').astype('int8');

      const npResult = runNumPy(`
import numpy as np
result = np.array([np.inf, -np.inf, np.nan], dtype=np.float64).astype(np.int8).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 inf/nan to uint8 matches NumPy', () => {
      const r = np.array([Infinity, -Infinity, NaN], 'float64').astype('uint8');

      const npResult = runNumPy(`
import numpy as np
result = np.array([np.inf, -np.inf, np.nan], dtype=np.float64).astype(np.uint8).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 very large value to int8 matches NumPy', () => {
      const r = np.array([1e20, -1e20], 'float64').astype('int8');

      const npResult = runNumPy(`
import numpy as np
result = np.array([1e20, -1e20], dtype=np.float64).astype(np.int8).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 large value to int32 saturates', () => {
      const r = np.array([3e9, -3e9, 1e20], 'float64').astype('int32');

      const npResult = runNumPy(`
import numpy as np
result = np.array([3e9, -3e9, 1e20], dtype=np.float64).astype(np.int32).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });

    it('float64 to int64 with special values', () => {
      const r = np.array([Infinity, -Infinity, NaN, 1e20], 'float64').astype('int64');

      const npResult = runNumPy(`
import numpy as np
result = np.array([np.inf, -np.inf, np.nan, 1e20]).astype(np.int64).tolist()
      `);

      expect(r.tolist().map(Number)).toEqual(npResult.value);
    });

    it('float64 truncates toward zero (not rounds)', () => {
      const r = np.array([1.9, -1.9, 2.5, -2.5], 'float64').astype('int32');

      const npResult = runNumPy(`
import numpy as np
result = np.array([1.9, -1.9, 2.5, -2.5]).astype(np.int32).tolist()
      `);

      expect(r.tolist()).toEqual(npResult.value);
    });
  });

  // ============================================================
  // MatMul Accumulation Overflow
  // ============================================================
  describe('matmul accumulation overflow', () => {
    it('float64 matmul overflows to inf', () => {
      const a = np.full([2, 2], 1e154, 'float64');
      const r = np.matmul(a, a);

      const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 1e154, dtype=np.float64)
result = float((a @ a)[0, 0])
      `);

      const val = (r.tolist() as number[][])[0]![0]!;
      expect(val).toBe(npResult.value);
      expect(val).toBe(Infinity);
    });

    it('float32 matmul overflows to inf', () => {
      const a = np.full([2, 2], 1e20, 'float32');
      const r = np.matmul(a, a);

      const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 1e20, dtype=np.float32)
result = float((a @ a)[0, 0])
      `);

      const val = (r.tolist() as number[][])[0]![0]!;
      expect(val).toBe(npResult.value);
    });

    it('int16 matmul wraps on overflow (matches NumPy dtype)', () => {
      const a = np.full([2, 2], 200, 'int16');
      const r = np.matmul(a, a);

      const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 200, dtype=np.int16)
r = a @ a
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

      expect(r.dtype).toBe('int16');
      expect(r.tolist()).toEqual(npResult.value.values);
    });

    it('int8 matmul wraps correctly', () => {
      const a = np.array(
        [
          [10, 20],
          [30, 40],
        ],
        'int8'
      );
      const b = np.array(
        [
          [5, 6],
          [7, 8],
        ],
        'int8'
      );
      const r = np.matmul(a, b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([[10, 20], [30, 40]], dtype=np.int8)
b = np.array([[5, 6], [7, 8]], dtype=np.int8)
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

      expect(r.dtype).toBe('int8');
      expect(r.tolist()).toEqual(npResult.value.values);
    });
  });

  // ============================================================
  // Bitwise Overflow
  // ============================================================
  describe('bitwise overflow', () => {
    it('int8 left_shift(1, 7) wraps to -128', () => {
      const r = np.left_shift(np.array([1], 'int8'), np.array([7], 'int8'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.left_shift(np.int8(1), np.int8(7)))
      `);

      expect(r.tolist()[0]).toBe(npResult.value);
    });

    it('uint8 left_shift(128, 1) wraps to 0', () => {
      const r = np.left_shift(np.array([128], 'uint8'), np.array([1], 'uint8'));

      const npResult = runNumPy(`
import numpy as np
result = int(np.left_shift(np.uint8(128), np.uint8(1)))
      `);

      expect(r.tolist()[0]).toBe(npResult.value);
    });
  });

  // ============================================================
  // Float Special Values
  // ============================================================
  describe('float special value propagation', () => {
    it('inf + inf = inf', () => {
      const a = np.array([Infinity], 'float64');
      expect(np.add(a, a).tolist()[0]).toBe(Infinity);
    });

    it('inf - inf = NaN', () => {
      const a = np.array([Infinity], 'float64');
      expect(np.subtract(a, a).tolist()[0]).toBeNaN();
    });

    it('0 * inf = NaN', () => {
      expect(
        np.multiply(np.array([0], 'float64'), np.array([Infinity], 'float64')).tolist()[0]
      ).toBeNaN();
    });

    it('inf / inf = NaN', () => {
      const a = np.array([Infinity], 'float64');
      expect(np.divide(a, a).tolist()[0]).toBeNaN();
    });

    it('NaN comparisons return false', () => {
      expect(np.equal(np.array([NaN], 'float64'), np.array([NaN], 'float64')).tolist()[0]).toBe(0);
    });
  });

  // ============================================================
  // Float Underflow
  // ============================================================
  describe('float underflow', () => {
    it('float32 underflow to zero', () => {
      const r = np.multiply(np.array([1e-45], 'float32'), np.array([1e-10], 'float32'));
      expect(r.tolist()[0]).toBe(0);
    });

    it('float64 underflow to zero', () => {
      const r = np.multiply(np.array([5e-324], 'float64'), np.array([0.1], 'float64'));
      expect(r.tolist()[0]).toBe(0);
    });
  });
});
