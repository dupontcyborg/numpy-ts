/**
 * TypeScript API Contract Tests
 *
 * Fast unit tests that validate our TypeScript API contracts without Python:
 * - Function signatures (parameter counts)
 * - Return types
 * - Metadata exports
 * - Basic functionality
 *
 * Note: For comprehensive NumPy API compatibility, see tests/validation/exports.numpy.test.ts
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src/index';

describe('API Exports - TypeScript Contracts', () => {
  describe('Package Metadata', () => {
    it('exports __version__', () => {
      expect(np.__version__).toBeDefined();
      expect(typeof np.__version__).toBe('string');
      expect(np.__version__).toMatch(/^\d+\.\d+\.\d+/);
    });
  });

  describe('Core Classes', () => {
    it('exports NDArray class', () => {
      expect(np.NDArray).toBeDefined();
      expect(typeof np.NDArray).toBe('function');
    });
  });

  describe('Function Signatures', () => {
    it('zeros(shape, dtype?)', () => {
      expect(np.zeros).toBeDefined();
      expect(typeof np.zeros).toBe('function');
      expect(np.zeros.length).toBe(1); // Required: shape

      // Test it works with various inputs
      expect(() => np.zeros([2, 3])).not.toThrow();
      expect(() => np.zeros([5])).not.toThrow();
    });

    it('ones(shape, dtype?)', () => {
      expect(np.ones).toBeDefined();
      expect(typeof np.ones).toBe('function');
      expect(np.ones.length).toBe(1); // Required: shape
    });

    it('array(object, dtype?)', () => {
      expect(np.array).toBeDefined();
      expect(typeof np.array).toBe('function');
      expect(np.array.length).toBe(2); // Parameters: object, dtype (optional)

      // Test nested arrays work
      expect(() =>
        np.array([
          [1, 2],
          [3, 4],
        ]),
      ).not.toThrow();
      expect(() => np.array([1, 2, 3])).not.toThrow();
    });

    it('arange(stop) or arange(start, stop, step?)', () => {
      expect(np.arange).toBeDefined();
      expect(typeof np.arange).toBe('function');
      expect(np.arange.length).toBeGreaterThanOrEqual(1);

      // Test various signatures
      expect(() => np.arange(5)).not.toThrow();
      expect(() => np.arange(0, 5)).not.toThrow();
      expect(() => np.arange(0, 5, 1)).not.toThrow();
    });

    it('linspace(start, stop, num?)', () => {
      expect(np.linspace).toBeDefined();
      expect(typeof np.linspace).toBe('function');
      expect(np.linspace.length).toBe(2); // Required: start, stop

      expect(() => np.linspace(0, 1)).not.toThrow();
      expect(() => np.linspace(0, 1, 50)).not.toThrow();
    });

    it('eye(n, m?, k?, dtype?)', () => {
      expect(np.eye).toBeDefined();
      expect(typeof np.eye).toBe('function');
      expect(np.eye.length).toBeGreaterThanOrEqual(1);

      expect(() => np.eye(3)).not.toThrow();
      expect(() => np.eye(3, 4)).not.toThrow();
      expect(() => np.eye(3, 4, 1)).not.toThrow();
    });
  });

  describe('Return Types', () => {
    it('creation functions return NDArray instances', () => {
      expect(np.zeros([2, 3])).toBeInstanceOf(np.NDArray);
      expect(np.ones([2, 3])).toBeInstanceOf(np.NDArray);
      expect(np.array([1, 2, 3])).toBeInstanceOf(np.NDArray);
      expect(np.arange(5)).toBeInstanceOf(np.NDArray);
      expect(np.linspace(0, 1, 5)).toBeInstanceOf(np.NDArray);
      expect(np.eye(3)).toBeInstanceOf(np.NDArray);
    });

    it('arithmetic methods return NDArray instances', () => {
      const arr = np.array([1, 2, 3]);
      expect(arr.add(1)).toBeInstanceOf(np.NDArray);
      expect(arr.subtract(1)).toBeInstanceOf(np.NDArray);
      expect(arr.multiply(2)).toBeInstanceOf(np.NDArray);
      expect(arr.divide(2)).toBeInstanceOf(np.NDArray);
    });

    it('reduction methods return numbers (scalars)', () => {
      const arr = np.array([1, 2, 3]);
      expect(typeof arr.sum()).toBe('number');
      expect(typeof arr.mean()).toBe('number');
      expect(typeof arr.max()).toBe('number');
      expect(typeof arr.min()).toBe('number');
    });

    it('matmul returns NDArray', () => {
      const A = np.array([
        [1, 2],
        [3, 4],
      ]);
      const B = np.array([
        [5, 6],
        [7, 8],
      ]);
      expect(A.matmul(B)).toBeInstanceOf(np.NDArray);
    });

    it('toArray returns nested JavaScript arrays', () => {
      const arr = np.array([
        [1, 2],
        [3, 4],
      ]);
      const result = arr.toArray();
      expect(Array.isArray(result)).toBe(true);
      expect(Array.isArray(result[0])).toBe(true);
    });

    it('toString returns string', () => {
      const arr = np.array([1, 2, 3]);
      expect(typeof arr.toString()).toBe('string');
    });
  });

  describe('NDArray Properties', () => {
    it('has correct property types', () => {
      const arr = np.zeros([2, 3]);

      // Properties return correct types
      expect(Array.isArray(arr.shape)).toBe(true);
      expect(typeof arr.ndim).toBe('number');
      expect(typeof arr.size).toBe('number');
      expect(typeof arr.dtype).toBe('string');
      expect(Array.isArray(arr.strides)).toBe(true);
      expect(arr.data).toBeDefined();
    });

    it('property values are consistent', () => {
      const arr = np.zeros([2, 3, 4]);

      expect(arr.shape).toEqual([2, 3, 4]);
      expect(arr.ndim).toBe(3);
      expect(arr.size).toBe(24); // 2 * 3 * 4
    });
  });

  describe('Method Contracts', () => {
    it('arithmetic methods accept scalar or NDArray', () => {
      const arr = np.array([1, 2, 3]);

      // Scalar operations
      expect(() => arr.add(5)).not.toThrow();
      expect(() => arr.subtract(2)).not.toThrow();
      expect(() => arr.multiply(3)).not.toThrow();
      expect(() => arr.divide(2)).not.toThrow();

      // Array operations
      const other = np.array([4, 5, 6]);
      expect(() => arr.add(other)).not.toThrow();
      expect(() => arr.subtract(other)).not.toThrow();
      expect(() => arr.multiply(other)).not.toThrow();
      expect(() => arr.divide(other)).not.toThrow();
    });

    it('reduction methods accept no arguments (for now)', () => {
      const arr = np.array([1, 2, 3]);

      // Currently these take no arguments (axis/keepdims not yet implemented)
      expect(() => arr.sum()).not.toThrow();
      expect(() => arr.mean()).not.toThrow();
      expect(() => arr.max()).not.toThrow();
      expect(() => arr.min()).not.toThrow();
    });

    it('matmul requires compatible shapes', () => {
      const A = np.array([
        [1, 2],
        [3, 4],
      ]);
      const B = np.array([
        [5, 6],
        [7, 8],
      ]);

      expect(() => A.matmul(B)).not.toThrow();
    });

    it('toArray works for all dimensions', () => {
      expect(() => np.array([1, 2, 3]).toArray()).not.toThrow();
      expect(() =>
        np
          .array([
            [1, 2],
            [3, 4],
          ])
          .toArray(),
      ).not.toThrow();
      expect(() => np.array([[[1]]]).toArray()).not.toThrow();
    });
  });

  describe('Basic Functionality Smoke Tests', () => {
    it('can create and manipulate arrays', () => {
      const a = np.array([1, 2, 3]);
      const b = a.add(10);
      const c = b.multiply(2);
      const result = c.toArray();

      expect(result).toEqual([22, 24, 26]);
    });

    it('can perform matrix operations', () => {
      const A = np.array([
        [1, 2],
        [3, 4],
      ]);
      const B = np.array([
        [2, 0],
        [1, 2],
      ]);
      const C = A.matmul(B);

      // Just verify it returns an NDArray with correct shape
      expect(C).toBeInstanceOf(np.NDArray);
      expect(C.shape).toEqual([2, 2]);
    });

    it('can compute reductions', () => {
      const arr = np.array([1, 2, 3, 4, 5]);

      expect(arr.sum()).toBe(15);
      expect(arr.mean()).toBe(3);
      expect(arr.max()).toBe(5);
      expect(arr.min()).toBe(1);
    });
  });
});
