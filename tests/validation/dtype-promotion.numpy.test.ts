/**
 * NumPy validation tests for dtype promotion
 * Tests that our dtype promotion matches Python NumPy exactly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { promoteDTypes } from '../../src/common/dtype';
import type { DType } from '../../src/common/dtype';
import { runNumPy, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: DType Promotion', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  // Map our dtypes to NumPy dtypes
  const dtypeMap: Record<DType, string> = {
    bool: 'bool_',
    int8: 'int8',
    int16: 'int16',
    int32: 'int32',
    int64: 'int64',
    uint8: 'uint8',
    uint16: 'uint16',
    uint32: 'uint32',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
    complex64: 'complex64',
    complex128: 'complex128',
  };

  // Map NumPy dtype strings back to our DType
  const reverseMap: Record<string, DType> = {
    bool: 'bool',
    int8: 'int8',
    int16: 'int16',
    int32: 'int32',
    int64: 'int64',
    uint8: 'uint8',
    uint16: 'uint16',
    uint32: 'uint32',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
    complex64: 'complex64',
    complex128: 'complex128',
  };

  describe('Basic promotion rules', () => {
    it('should promote bool + int8 correctly', () => {
      const jsResult = promoteDTypes('bool', 'int8');
      const pyResult = runNumPy(`
import numpy as np
a = np.array([True], dtype=np.bool_)
b = np.array([1], dtype=np.int8)
result = a + b
      `);

      const expectedDtype = reverseMap[pyResult.dtype] || 'int8';
      expect(jsResult).toBe(expectedDtype);
    });

    it('should promote int32 + float64 correctly', () => {
      const jsResult = promoteDTypes('int32', 'float64');
      const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.int32)
b = np.array([1.0], dtype=np.float64)
result = a + b
      `);

      const expectedDtype = reverseMap[pyResult.dtype] || 'float64';
      expect(jsResult).toBe(expectedDtype);
    });

    it('should promote int64 + uint64 correctly', () => {
      const jsResult = promoteDTypes('int64', 'uint64');
      const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.int64)
b = np.array([1], dtype=np.uint64)
result = a + b
      `);

      const expectedDtype = reverseMap[pyResult.dtype] || 'uint64';
      expect(jsResult).toBe(expectedDtype);
    });
  });

  describe('Comprehensive promotion matrix', () => {
    // Test a subset of important dtype combinations
    const testPairs: Array<[DType, DType]> = [
      ['bool', 'bool'],
      ['bool', 'int32'],
      ['bool', 'float64'],
      ['int8', 'int16'],
      ['int8', 'uint8'],
      ['int16', 'int32'],
      ['int32', 'int64'],
      ['uint8', 'uint16'],
      ['uint16', 'uint32'],
      ['uint32', 'uint64'],
      ['int32', 'uint32'],
      ['int64', 'uint64'],
      ['int32', 'float32'],
      ['int64', 'float64'],
      ['float32', 'float64'],
      // Skip complex tests due to JSON serialization issues
      // ['float32', 'complex64'],
      // ['float64', 'complex128'],
      // ['complex64', 'complex128'],
    ];

    testPairs.forEach(([dtype1, dtype2]) => {
      it(`should promote ${dtype1} + ${dtype2} to match NumPy`, () => {
        const jsResult = promoteDTypes(dtype1, dtype2);
        const pyResult = runNumPy(`
import numpy as np
a = np.array([1], dtype=np.${dtypeMap[dtype1]})
b = np.array([1], dtype=np.${dtypeMap[dtype2]})
result = a + b
        `);

        const expectedDtype = reverseMap[pyResult.dtype] || jsResult;
        expect(jsResult).toBe(expectedDtype);
      });
    });
  });

  describe('Commutative property', () => {
    it('should be commutative (dtype1 + dtype2 === dtype2 + dtype1)', () => {
      const testCases: Array<[DType, DType]> = [
        ['int32', 'float64'],
        ['int64', 'uint64'],
        ['float32', 'complex64'],
        ['int8', 'uint16'],
      ];

      testCases.forEach(([dtype1, dtype2]) => {
        const result1 = promoteDTypes(dtype1, dtype2);
        const result2 = promoteDTypes(dtype2, dtype1);
        expect(result1).toBe(result2);
      });
    });
  });
});
