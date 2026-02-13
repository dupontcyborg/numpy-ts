/**
 * NumPy validation tests for integer overflow behavior
 * Validates that overflow wrapping matches NumPy exactly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';
import type { DType } from '../../src/common/dtype';

describe('NumPy Validation: Integer Overflow Behavior', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('int8 overflow (range: -128 to 127)', () => {
    it('positive overflow wraps to negative (matches NumPy)', () => {
      const arr = array([127], 'int8');
      const result = arr.add(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([127], dtype=np.int8)
result = arr + 1
      `);

      expect(result.dtype).toBe('int8');
      expect(result.get([0])).toBe(-128);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('large positive overflow wraps correctly (matches NumPy)', () => {
      const arr = array([127], 'int8');
      const result = arr.add(10);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([127], dtype=np.int8)
result = arr + 10
      `);

      expect(result.get([0])).toBe(-119);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative underflow wraps to positive (matches NumPy)', () => {
      const arr = array([-128], 'int8');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([-128], dtype=np.int8)
result = arr - 1
      `);

      expect(result.get([0])).toBe(127);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('multiplication overflow wraps (matches NumPy)', () => {
      const arr = array([100], 'int8');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([100], dtype=np.int8)
result = arr * 2
      `);

      expect(result.get([0])).toBe(-56);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('array-to-array overflow (matches NumPy)', () => {
      const a = array([120, 100, -120], 'int8');
      const b = array([10, 30, -10], 'int8');
      const result = a.add(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([120, 100, -120], dtype=np.int8)
b = np.array([10, 30, -10], dtype=np.int8)
result = a + b
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      expect(result.get([1])).toBe(npResult.value[1]);
      expect(result.get([2])).toBe(npResult.value[2]);
    });
  });

  describe('uint8 overflow (range: 0 to 255)', () => {
    it('positive overflow wraps to zero (matches NumPy)', () => {
      const arr = array([255], 'uint8');
      const result = arr.add(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([255], dtype=np.uint8)
result = arr + 1
      `);

      expect(result.get([0])).toBe(0);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('large positive overflow wraps correctly (matches NumPy)', () => {
      const arr = array([255], 'uint8');
      const result = arr.add(10);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([255], dtype=np.uint8)
result = arr + 10
      `);

      expect(result.get([0])).toBe(9);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative wraps to high values (matches NumPy)', () => {
      const arr = array([0], 'uint8');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([0], dtype=np.uint8)
result = arr - 1
      `);

      expect(result.get([0])).toBe(255);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('multiplication overflow wraps (matches NumPy)', () => {
      const arr = array([200], 'uint8');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([200], dtype=np.uint8)
result = arr * 2
      `);

      expect(result.get([0])).toBe(144);
      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });

  describe('int16 overflow (range: -32768 to 32767)', () => {
    it('positive overflow wraps to negative (matches NumPy)', () => {
      const arr = array([32767], 'int16');
      const result = arr.add(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([32767], dtype=np.int16)
result = arr + 1
      `);

      expect(result.get([0])).toBe(-32768);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative underflow wraps to positive (matches NumPy)', () => {
      const arr = array([-32768], 'int16');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([-32768], dtype=np.int16)
result = arr - 1
      `);

      expect(result.get([0])).toBe(32767);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('multiplication overflow wraps (matches NumPy)', () => {
      const arr = array([20000], 'int16');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([20000], dtype=np.int16)
result = arr * 2
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });

  describe('uint16 overflow (range: 0 to 65535)', () => {
    it('positive overflow wraps to zero (matches NumPy)', () => {
      const arr = array([65535], 'uint16');
      const result = arr.add(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([65535], dtype=np.uint16)
result = arr + 1
      `);

      expect(result.get([0])).toBe(0);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative wraps to high values (matches NumPy)', () => {
      const arr = array([0], 'uint16');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([0], dtype=np.uint16)
result = arr - 1
      `);

      expect(result.get([0])).toBe(65535);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('multiplication overflow wraps (matches NumPy)', () => {
      const arr = array([50000], 'uint16');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([50000], dtype=np.uint16)
result = arr * 2
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });

  describe('int32 overflow (range: -2147483648 to 2147483647)', () => {
    it('positive overflow wraps to negative (matches NumPy)', () => {
      const arr = array([2147483647], 'int32');
      const result = arr.add(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([2147483647], dtype=np.int32)
result = arr + 1
      `);

      expect(result.get([0])).toBe(-2147483648);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative underflow wraps to positive (matches NumPy)', () => {
      const arr = array([-2147483648], 'int32');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([-2147483648], dtype=np.int32)
result = arr - 1
      `);

      expect(result.get([0])).toBe(2147483647);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('multiplication overflow wraps (matches NumPy)', () => {
      const arr = array([2000000000], 'int32');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([2000000000], dtype=np.int32)
result = arr * 2
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });

  describe('uint32 overflow (range: 0 to 4294967295)', () => {
    it('positive overflow wraps to zero (matches NumPy)', () => {
      const arr = array([4294967295], 'uint32');
      const result = arr.add(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([4294967295], dtype=np.uint32)
result = arr + 1
      `);

      expect(result.get([0])).toBe(0);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative wraps to high values (matches NumPy)', () => {
      const arr = array([0], 'uint32');
      const result = arr.subtract(1);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([0], dtype=np.uint32)
result = arr - 1
      `);

      expect(result.get([0])).toBe(4294967295);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('multiplication overflow wraps (matches NumPy)', () => {
      const arr = array([3000000000], 'uint32');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([3000000000], dtype=np.uint32)
result = arr * 2
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });

  describe('int64 overflow (BigInt type)', () => {
    it('positive overflow wraps to negative (matches NumPy)', () => {
      const arr = array([9007199254740991], 'int64'); // Max safe integer
      const result = arr.add(10);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([9007199254740991], dtype=np.int64)
result = arr + 10
      `);

      // Compare values (BigInt to Number conversion for comparison)
      expect(result.dtype).toBe('int64');
      expect(npResult.dtype).toBe('int64');
      expect(typeof result.data[0]).toBe('bigint');
      expect(Number(result.data[0])).toBe(npResult.value[0]);
    });

    it('handles large int64 values (matches NumPy)', () => {
      const arr = array([1000000000000], 'int64');
      const result = arr.add(1000000000000);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([1000000000000], dtype=np.int64)
result = arr + 1000000000000
      `);

      expect(result.dtype).toBe('int64');
      expect(Number(result.data[0])).toBe(2000000000000);
      expect(Number(result.data[0])).toBe(npResult.value[0]);
    });
  });

  describe('uint64 overflow (BigInt type)', () => {
    it('handles large uint64 values (matches NumPy)', () => {
      const arr = array([1000000000000], 'uint64');
      const result = arr.add(1000000000000);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([1000000000000], dtype=np.uint64)
result = arr + 1000000000000
      `);

      expect(result.dtype).toBe('uint64');
      expect(Number(result.data[0])).toBe(2000000000000);
      expect(Number(result.data[0])).toBe(npResult.value[0]);
    });
  });

  describe('Mixed overflow scenarios', () => {
    it('complex calculation with multiple overflows (matches NumPy)', () => {
      const arr = array([100, 127, -100, -128], 'int8');
      const result = arr.add(50).multiply(2).subtract(10);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([100, 127, -100, -128], dtype=np.int8)
result = ((arr + 50) * 2) - 10
      `);

      expect(result.get([0])).toBe(npResult.value[0]);
      expect(result.get([1])).toBe(npResult.value[1]);
      expect(result.get([2])).toBe(npResult.value[2]);
      expect(result.get([3])).toBe(npResult.value[3]);
    });

    it('overflow with different dtypes in promotion (matches NumPy)', () => {
      // When mixing dtypes, promotion happens first, then overflow in promoted type
      const a = array([127], 'int8');
      const b = array([1], 'int16');
      const result = a.add(b);

      const npResult = runNumPy(`
import numpy as np
a = np.array([127], dtype=np.int8)
b = np.array([1], dtype=np.int16)
result = a + b
      `);

      // Should promote to int16, so no overflow
      expect(result.dtype).toBe('int16');
      expect(result.get([0])).toBe(128);
      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });

  describe('Edge case overflow scenarios', () => {
    const intTypes: Array<{ dtype: DType; min: number; max: number }> = [
      { dtype: 'int8', min: -128, max: 127 },
      { dtype: 'int16', min: -32768, max: 32767 },
      { dtype: 'int32', min: -2147483648, max: 2147483647 },
      { dtype: 'uint8', min: 0, max: 255 },
      { dtype: 'uint16', min: 0, max: 65535 },
      { dtype: 'uint32', min: 0, max: 4294967295 },
    ];

    for (const { dtype, min, max } of intTypes) {
      it(`${dtype}: max + max wraps correctly (matches NumPy)`, () => {
        const arr = array([max], dtype);
        const result = arr.add(max);

        const npResult = runNumPy(`
import numpy as np
arr = np.array([${max}], dtype=np.${dtype})
result = arr + ${max}
        `);

        expect(result.get([0])).toBe(npResult.value[0]);
      });

      it(`${dtype}: min - min wraps correctly (matches NumPy)`, () => {
        const arr = array([min], dtype);
        const result = arr.subtract(min);

        const npResult = runNumPy(`
import numpy as np
arr = np.array([${min}], dtype=np.${dtype})
result = arr - ${min}
        `);

        expect(result.get([0])).toBe(npResult.value[0]);
      });
    }
  });

  describe('Float overflow produces Infinity (IEEE 754 behavior)', () => {
    it('float32 overflow produces Infinity (matches NumPy)', () => {
      const arr = array([3.4e38], 'float32');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
import warnings
arr = np.array([3.4e38], dtype=np.float32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr * 2
      `);

      expect(result.dtype).toBe('float32');
      expect(result.get([0])).toBe(Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('float32 large overflow produces Infinity (matches NumPy)', () => {
      const arr = array([3.4e38], 'float32');
      const result = arr.multiply(10);

      const npResult = runNumPy(`
import numpy as np
import warnings
arr = np.array([3.4e38], dtype=np.float32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr * 10
      `);

      expect(result.get([0])).toBe(Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('float64 overflow produces Infinity (matches NumPy)', () => {
      const arr = array([1.7e308], 'float64');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
import warnings
arr = np.array([1.7e308], dtype=np.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr * 2
      `);

      expect(result.dtype).toBe('float64');
      expect(result.get([0])).toBe(Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('float32 underflow produces zero (matches NumPy)', () => {
      const arr = array([1e-40], 'float32');
      const result = arr.divide(1e10);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([1e-40], dtype=np.float32)
result = arr / 1e10
      `);

      expect(result.get([0])).toBe(0);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('float64 underflow produces zero (matches NumPy)', () => {
      const arr = array([1e-320], 'float64');
      const result = arr.divide(1e10);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([1e-320], dtype=np.float64)
result = arr / 1e10
      `);

      expect(result.get([0])).toBe(0);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('operations on Infinity stay as Infinity (matches NumPy)', () => {
      const arr = array([Infinity], 'float32');
      const result = arr.add(1).multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([np.inf], dtype=np.float32)
result = (arr + 1) * 2
      `);

      expect(result.get([0])).toBe(Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('negative overflow produces -Infinity (matches NumPy)', () => {
      const arr = array([-3.4e38], 'float32');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
import warnings
arr = np.array([-3.4e38], dtype=np.float32)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = arr * 2
      `);

      expect(result.get([0])).toBe(-Infinity);
      expect(result.get([0])).toBe(npResult.value[0]);
    });

    it('finite result from large values (matches NumPy)', () => {
      const arr = array([1e30], 'float32');
      const result = arr.multiply(2);

      const npResult = runNumPy(`
import numpy as np
arr = np.array([1e30], dtype=np.float32)
result = arr * 2
      `);

      // Should be finite and match NumPy
      expect(Number.isFinite(result.get([0]))).toBe(true);
      expect(result.get([0])).toBe(npResult.value[0]);
    });
  });
});
