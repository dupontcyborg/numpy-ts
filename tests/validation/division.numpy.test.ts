/**
 * NumPy validation tests for division operations
 * Validates division behavior including type promotion and division by zero
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { ones, zeros, array } from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';
import type { DType } from '../../src/common/dtype';

describe('NumPy Validation: Division Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('Integer division promotes to float64', () => {
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
      it(`${dtype} / scalar promotes to float64 (matches NumPy)`, () => {
        const arr = array([4, 6, 8], dtype);
        const result = arr.divide(2);

        const npResult = runNumPy(`
import numpy as np
arr = np.array([4, 6, 8], dtype=np.${dtype})
result = arr / 2
        `);

        expect(result.dtype).toBe('float64');
        expect(npResult.dtype).toBe('float64');
        expect(result.get([0])).toBeCloseTo(npResult.value[0], 10);
      });

      it(`${dtype} array / ${dtype} array promotes to float64 (matches NumPy)`, () => {
        const a = array([4, 6, 8], dtype);
        const b = array([2, 2, 2], dtype);
        const result = a.divide(b);

        const npResult = runNumPy(`
import numpy as np
a = np.array([4, 6, 8], dtype=np.${dtype})
b = np.array([2, 2, 2], dtype=np.${dtype})
result = a / b
        `);

        expect(result.dtype).toBe('float64');
        expect(npResult.dtype).toBe('float64');
      });
    }
  });

  describe('Float division preserves dtype', () => {
    it('float32 / scalar stays float32 (matches NumPy)', () => {
      const arr = array([4, 6, 8], 'float32');
      const result = arr.divide(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([4, 6, 8], dtype=np.float32)
result = arr / 2
      `);

      expect(result.dtype).toBe('float32');
      expect(npResult.dtype).toBe('float32');
    });

    it('float64 / scalar stays float64 (matches NumPy)', () => {
      const arr = array([4, 6, 8], 'float64');
      const result = arr.divide(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([4, 6, 8], dtype=np.float64)
result = arr / 2
      `);

      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
    });
  });

  describe('Division by zero behavior matches NumPy', () => {
    it('float64: positive / 0 returns Infinity (matches NumPy)', () => {
      const arr = ones([2], 'float64');
      const result = arr.divide(0);

      // NumPy also returns inf (with warning)
      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(Infinity);
    });

    it('float64: 0 / 0 returns NaN (matches NumPy)', () => {
      const arr = zeros([2], 'float64');
      const result = arr.divide(0);

      // NumPy also returns nan (with warning)
      expect(result.dtype).toBe('float64');
      expect(Number.isNaN(result.get([0]))).toBe(true);
    });

    it('int32 / 0 promotes to float64 and returns Infinity (matches NumPy)', () => {
      const arr = ones([2], 'int32');
      const result = arr.divide(0);

      // NumPy promotes to float64 and returns inf
      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(Infinity);
    });

    it('int64 / 0 promotes to float64 and returns Infinity (matches NumPy)', () => {
      const arr = ones([2], 'int64');
      const result = arr.divide(0);

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(Infinity);
    });
  });

  describe('Array-to-array division matches NumPy', () => {
    it('int32 array / array with zeros (matches NumPy promotion)', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([1, 0, 1], 'int32');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
import warnings
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1, 0, 1], dtype=np.int32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = a / b
      `);

      // NumPy promotes int32/int32 division to float64
      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');

      // Check our result values match NumPy exactly (including Infinity)
      expect(result.get([0])).toBe(npResult.value[0]); // 1 / 1 = 1
      expect(result.get([1])).toBe(npResult.value[1]); // 2 / 0 = Inf
      expect(result.get([2])).toBe(npResult.value[2]); // 3 / 1 = 3
    });

    it('float64 array / array without zeros matches NumPy', () => {
      const a = array([10, 20, 30], 'float64');
      const b = array([2, 4, 5], 'float64');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([10, 20, 30], dtype=np.float64)
b = np.array([2, 4, 5], dtype=np.float64)
result = a / b
      `);

      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
      expect(result.get([0])).toBeCloseTo(npResult.value[0], 10);
      expect(result.get([1])).toBeCloseTo(npResult.value[1], 10);
      expect(result.get([2])).toBeCloseTo(npResult.value[2], 10);
    });
  });

  describe('Mixed dtype division matches NumPy', () => {
    it('int32 / float64 matches NumPy promotion', () => {
      const a = array([4, 6, 8], 'int32');
      const b = array([2, 2, 2], 'float64');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([4, 6, 8], dtype=np.int32)
b = np.array([2, 2, 2], dtype=np.float64)
result = a / b
      `);

      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
    });

    it('float32 / int16 matches NumPy promotion', () => {
      const a = array([4, 6, 8], 'float32');
      const b = array([2, 2, 2], 'int16');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([4, 6, 8], dtype=np.float32)
b = np.array([2, 2, 2], dtype=np.int16)
result = a / b
      `);

      // Result dtype should match NumPy
      expect(result.dtype).toBe(npResult.dtype);
    });
  });

  describe('Broadcasting with division matches NumPy', () => {
    it('2D / 1D broadcasting matches NumPy', () => {
      const a = array(
        [
          [4, 6],
          [8, 10],
        ],
        'int32'
      );
      const b = array([2, 3], 'int32');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([[4, 6], [8, 10]], dtype=np.int32)
b = np.array([2, 3], dtype=np.int32)
result = a / b
      `);

      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
      expect(result.shape).toEqual([2, 2]);
      expect(result.get([0, 0])).toBeCloseTo(npResult.value[0][0], 10);
      expect(result.get([0, 1])).toBeCloseTo(npResult.value[0][1], 10);
    });
  });

  describe('Edge cases match NumPy', () => {
    it('bool array division matches NumPy', () => {
      const a = array([1, 0, 1], 'bool');
      const b = array([1, 1, 1], 'bool');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([1, 0, 1], dtype=np.bool_)
b = np.array([1, 1, 1], dtype=np.bool_)
result = a / b
      `);

      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
    });

    it('large integer values division matches NumPy', () => {
      const a = array([1000000, 2000000], 'int32');
      const b = array([3, 4], 'int32');
      const result = a.divide(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([1000000, 2000000], dtype=np.int32)
b = np.array([3, 4], dtype=np.int32)
result = a / b
      `);

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBeCloseTo(npResult.value[0], 5);
      expect(result.get([1])).toBeCloseTo(npResult.value[1], 5);
    });
  });
});
