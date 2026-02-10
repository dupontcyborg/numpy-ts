/**
 * NumPy validation tests for dtype retention
 * Compares our dtype retention behavior against actual NumPy
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { ones } from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';
import type { DType } from '../../src/common/dtype';

describe('NumPy Validation: DType Retention', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  // Map our dtypes to NumPy dtypes
  const dtypeMap: Record<DType, string> = {
    float64: 'float64',
    float32: 'float32',
    int64: 'int64',
    int32: 'int32',
    int16: 'int16',
    int8: 'int8',
    uint64: 'uint64',
    uint32: 'uint32',
    uint16: 'uint16',
    uint8: 'uint8',
    bool: 'bool',
  };

  const testDTypes: DType[] = ['float64', 'float32', 'int32', 'int16', 'uint8'];

  describe('Shape operations match NumPy dtype behavior', () => {
    it('reshape() preserves dtype like NumPy', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const reshaped = arr.reshape([3, 2]);

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr.reshape(3, 2)
        `);

        expect(reshaped.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }
    });

    it('flatten() preserves dtype like NumPy', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const flat = arr.flatten();

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr.flatten()
        `);

        expect(flat.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }
    });

    it('transpose() preserves dtype like NumPy', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const transposed = arr.transpose();

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr.T
        `);

        expect(transposed.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }
    });
  });

  describe('Arithmetic operations match NumPy dtype behavior', () => {
    it('scalar addition preserves dtype like NumPy', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const result = arr.add(5);

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr + 5
        `);

        expect(result.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }
    });

    it('array addition with type promotion matches NumPy', () => {
      const a = ones([2], 'int32');
      const b = ones([2], 'float64');
      const result = a.add(b);

      const npResult = runNumPy(`
a = np.ones(2, dtype=np.int32)
b = np.ones(2, dtype=np.float64)
result = a + b
      `);

      // Both should promote to float64
      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
    });
  });

  describe('Reductions match NumPy dtype behavior', () => {
    it('sum() preserves dtype like NumPy', () => {
      // Note: NumPy may promote integer types during sum to prevent overflow
      // We test that dtypes are preserved for float types
      const floatDTypes: DType[] = ['float64', 'float32'];
      for (const dtype of floatDTypes) {
        const arr = ones([2, 3], dtype);
        const result = arr.sum(0);

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr.sum(axis=0)
        `);

        expect(result.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }

      // For integer types, just verify we get an integer result
      const arr = ones([2, 3], 'int32');
      const result = arr.sum(0);
      expect(result.dtype).toMatch(/int/);
    });

    it('mean() converts integer types to float64 like NumPy', () => {
      const arr = ones([2, 3], 'int32');
      const result = arr.mean(0);

      const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.int32)
result = arr.mean(axis=0)
      `);

      // Both should convert to float64
      expect(result.dtype).toBe('float64');
      expect(npResult.dtype).toBe('float64');
    });

    it('mean() preserves float dtype like NumPy', () => {
      const arr = ones([2, 3], 'float32');
      const result = arr.mean(0);

      const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.float32)
result = arr.mean(axis=0)
      `);

      // Float32 should stay float32
      expect(result.dtype).toBe('float32');
      expect(npResult.dtype).toBe('float32');
    });

    it('max() preserves dtype like NumPy', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const result = arr.max(0);

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr.max(axis=0)
        `);

        expect(result.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }
    });
  });

  describe('Comparisons return bool like NumPy', () => {
    it('comparison operations return bool dtype', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const result = arr.greater(0);

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr > 0
        `);

        expect(result.dtype).toBe('bool');
        expect(npResult.dtype).toBe('bool');
      }
    });
  });

  describe('Copy operations preserve dtype like NumPy', () => {
    it('copy() preserves dtype', () => {
      for (const dtype of testDTypes) {
        const arr = ones([2, 3], dtype);
        const copied = arr.copy();

        const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.${dtypeMap[dtype]})
result = arr.copy()
        `);

        expect(copied.dtype).toBe(dtype);
        expect(npResult.dtype).toContain(dtype);
      }
    });

    it('astype() converts dtype like NumPy', () => {
      const arr = ones([2, 3], 'float64');
      const converted = arr.astype('int32');

      const npResult = runNumPy(`
arr = np.ones((2, 3), dtype=np.float64)
result = arr.astype(np.int32)
      `);

      expect(converted.dtype).toBe('int32');
      expect(npResult.dtype).toBe('int32');
    });
  });
});
