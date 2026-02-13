/**
 * Comprehensive dtype promotion matrix tests
 * Tests all 11×11 dtype combinations to ensure correct promotion behavior
 */

import { describe, it, expect } from 'vitest';
import { ones } from '../../src';
import { promoteDTypes } from '../../src/common/dtype';
import type { DType } from '../../src/common/dtype';

describe('Complete DType Promotion Matrix', () => {
  // All supported dtypes (11 types)
  const allDTypes: DType[] = [
    'bool',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float32',
    'float64',
  ];

  // Expected promotion results for all combinations
  // Based on NumPy's type promotion rules
  const expectedPromotions: Record<DType, Record<DType, DType>> = {
    bool: {
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
    },
    int8: {
      bool: 'int8',
      int8: 'int8',
      int16: 'int16',
      int32: 'int32',
      int64: 'int64',
      uint8: 'int16',
      uint16: 'int32',
      uint32: 'int64',
      uint64: 'float64',
      float32: 'float32',
      float64: 'float64',
    },
    int16: {
      bool: 'int16',
      int8: 'int16',
      int16: 'int16',
      int32: 'int32',
      int64: 'int64',
      uint8: 'int16',
      uint16: 'int32',
      uint32: 'int64',
      uint64: 'float64',
      float32: 'float32',
      float64: 'float64',
    },
    int32: {
      bool: 'int32',
      int8: 'int32',
      int16: 'int32',
      int32: 'int32',
      int64: 'int64',
      uint8: 'int32',
      uint16: 'int32',
      uint32: 'int64',
      uint64: 'float64',
      float32: 'float64',
      float64: 'float64',
    },
    int64: {
      bool: 'int64',
      int8: 'int64',
      int16: 'int64',
      int32: 'int64',
      int64: 'int64',
      uint8: 'int64',
      uint16: 'int64',
      uint32: 'int64',
      uint64: 'float64',
      float32: 'float64',
      float64: 'float64',
    },
    uint8: {
      bool: 'uint8',
      int8: 'int16',
      int16: 'int16',
      int32: 'int32',
      int64: 'int64',
      uint8: 'uint8',
      uint16: 'uint16',
      uint32: 'uint32',
      uint64: 'uint64',
      float32: 'float32',
      float64: 'float64',
    },
    uint16: {
      bool: 'uint16',
      int8: 'int32',
      int16: 'int32',
      int32: 'int32',
      int64: 'int64',
      uint8: 'uint16',
      uint16: 'uint16',
      uint32: 'uint32',
      uint64: 'uint64',
      float32: 'float32',
      float64: 'float64',
    },
    uint32: {
      bool: 'uint32',
      int8: 'int64',
      int16: 'int64',
      int32: 'int64',
      int64: 'int64',
      uint8: 'uint32',
      uint16: 'uint32',
      uint32: 'uint32',
      uint64: 'uint64',
      float32: 'float64',
      float64: 'float64',
    },
    uint64: {
      bool: 'uint64',
      int8: 'float64',
      int16: 'float64',
      int32: 'float64',
      int64: 'float64',
      uint8: 'uint64',
      uint16: 'uint64',
      uint32: 'uint64',
      uint64: 'uint64',
      float32: 'float64',
      float64: 'float64',
    },
    float32: {
      bool: 'float32',
      int8: 'float32',
      int16: 'float32',
      int32: 'float64',
      int64: 'float64',
      uint8: 'float32',
      uint16: 'float32',
      uint32: 'float64',
      uint64: 'float64',
      float32: 'float32',
      float64: 'float64',
    },
    float64: {
      bool: 'float64',
      int8: 'float64',
      int16: 'float64',
      int32: 'float64',
      int64: 'float64',
      uint8: 'float64',
      uint16: 'float64',
      uint32: 'float64',
      uint64: 'float64',
      float32: 'float64',
      float64: 'float64',
    },
  };

  describe('promoteDTypes() function - all 121 combinations', () => {
    for (const dtype1 of allDTypes) {
      for (const dtype2 of allDTypes) {
        it(`promotes ${dtype1} + ${dtype2} correctly`, () => {
          const result = promoteDTypes(dtype1, dtype2);
          const expected = expectedPromotions[dtype1][dtype2];
          expect(result).toBe(expected);
        });
      }
    }
  });

  describe('Array arithmetic promotion - all 121 combinations', () => {
    for (const dtype1 of allDTypes) {
      for (const dtype2 of allDTypes) {
        it(`${dtype1} array + ${dtype2} array promotes to ${expectedPromotions[dtype1][dtype2]}`, () => {
          const a = ones([2, 2], dtype1);
          const b = ones([2, 2], dtype2);
          const result = a.add(b);
          const expected = expectedPromotions[dtype1][dtype2];
          expect(result.dtype).toBe(expected);
        });
      }
    }
  });

  describe('Commutative property - promoteDTypes(a, b) === promoteDTypes(b, a)', () => {
    for (const dtype1 of allDTypes) {
      for (const dtype2 of allDTypes) {
        it(`${dtype1} + ${dtype2} === ${dtype2} + ${dtype1}`, () => {
          const result1 = promoteDTypes(dtype1, dtype2);
          const result2 = promoteDTypes(dtype2, dtype1);
          expect(result1).toBe(result2);
        });
      }
    }
  });

  describe('Idempotent property - promoteDTypes(a, a) === a', () => {
    for (const dtype of allDTypes) {
      it(`${dtype} + ${dtype} === ${dtype}`, () => {
        const result = promoteDTypes(dtype, dtype);
        expect(result).toBe(dtype);
      });
    }
  });

  describe('Promotion hierarchy - float > int > uint > bool', () => {
    it('float64 is the "strongest" type', () => {
      for (const dtype of allDTypes) {
        if (dtype === 'float64') continue;
        const result = promoteDTypes('float64', dtype);
        expect(result).toBe('float64');
      }
    });

    it('bool is the "weakest" type', () => {
      for (const dtype of allDTypes) {
        const result = promoteDTypes('bool', dtype);
        expect(result).toBe(dtype);
      }
    });

    it('signed + unsigned of same size promotes to larger signed', () => {
      expect(promoteDTypes('int8', 'uint8')).toBe('int16');
      expect(promoteDTypes('int16', 'uint16')).toBe('int32');
      expect(promoteDTypes('int32', 'uint32')).toBe('int64');
    });

    it('int64 + uint64 promotes to float64 (no larger int available)', () => {
      expect(promoteDTypes('int64', 'uint64')).toBe('float64');
    });

    it('small int + float32 promotes to float32, large int + float32 promotes to float64', () => {
      // Small integers (8, 16 bit) can be represented in float32
      expect(promoteDTypes('int8', 'float32')).toBe('float32');
      expect(promoteDTypes('int16', 'float32')).toBe('float32');
      expect(promoteDTypes('uint8', 'float32')).toBe('float32');
      expect(promoteDTypes('uint16', 'float32')).toBe('float32');

      // Large integers (32, 64 bit) need float64 for precision
      expect(promoteDTypes('int32', 'float32')).toBe('float64');
      expect(promoteDTypes('int64', 'float32')).toBe('float64');
      expect(promoteDTypes('uint32', 'float32')).toBe('float64');
      expect(promoteDTypes('uint64', 'float32')).toBe('float64');
    });

    it('float32 + float32 stays float32', () => {
      expect(promoteDTypes('float32', 'float32')).toBe('float32');
    });
  });

  describe('Specific promotion rules', () => {
    it('same signedness promotes to larger size', () => {
      // Signed integers
      expect(promoteDTypes('int8', 'int16')).toBe('int16');
      expect(promoteDTypes('int8', 'int32')).toBe('int32');
      expect(promoteDTypes('int16', 'int32')).toBe('int32');
      expect(promoteDTypes('int32', 'int64')).toBe('int64');

      // Unsigned integers
      expect(promoteDTypes('uint8', 'uint16')).toBe('uint16');
      expect(promoteDTypes('uint8', 'uint32')).toBe('uint32');
      expect(promoteDTypes('uint16', 'uint32')).toBe('uint32');
      expect(promoteDTypes('uint32', 'uint64')).toBe('uint64');
    });

    it('mixed signedness: signed type can hold unsigned if larger', () => {
      // When signed type is larger, it can hold the unsigned type
      expect(promoteDTypes('int16', 'uint8')).toBe('int16');
      expect(promoteDTypes('int32', 'uint8')).toBe('int32');
      expect(promoteDTypes('int32', 'uint16')).toBe('int32');
      expect(promoteDTypes('int64', 'uint8')).toBe('int64');
      expect(promoteDTypes('int64', 'uint16')).toBe('int64');
      expect(promoteDTypes('int64', 'uint32')).toBe('int64');

      // When unsigned type is larger, need bigger signed type
      expect(promoteDTypes('int8', 'uint16')).toBe('int32');
      expect(promoteDTypes('int8', 'uint32')).toBe('int64');
      expect(promoteDTypes('int16', 'uint32')).toBe('int64');
    });
  });

  describe('Edge cases', () => {
    it('handles all operations with bool', () => {
      const boolArr = ones([2], 'bool');

      expect(boolArr.add(ones([2], 'int8')).dtype).toBe('int8');
      expect(boolArr.subtract(ones([2], 'uint16')).dtype).toBe('uint16');
      expect(boolArr.multiply(ones([2], 'float32')).dtype).toBe('float32');
      // Division promotes to float64 (NumPy behavior)
      expect(boolArr.divide(ones([2], 'int64')).dtype).toBe('float64');
    });

    it('handles operations between extreme types', () => {
      const smallest = ones([2], 'bool');
      const largest = ones([2], 'float64');
      expect(smallest.add(largest).dtype).toBe('float64');
    });

    it('handles BigInt types correctly', () => {
      const int64Arr = ones([2], 'int64');
      const uint64Arr = ones([2], 'uint64');

      expect(int64Arr.add(int64Arr).dtype).toBe('int64');
      expect(uint64Arr.add(uint64Arr).dtype).toBe('uint64');
      expect(int64Arr.add(uint64Arr).dtype).toBe('float64');
    });
  });
});
