/**
 * Unit tests for broadcasting utilities
 */

import { describe, it, expect } from 'vitest';
import {
  computeBroadcastShape,
  areBroadcastable,
  broadcastArrays,
  broadcastErrorMessage,
} from '../../src/common/broadcasting';
import { array, zeros, ones, broadcast_shapes } from '../../src';

describe('Broadcasting utilities', () => {
  describe('computeBroadcastShape', () => {
    it('returns shape for single array', () => {
      expect(computeBroadcastShape([[3, 4]])).toEqual([3, 4]);
    });

    it('returns empty array for no shapes', () => {
      expect(computeBroadcastShape([])).toEqual([]);
    });

    it('broadcasts scalar (0d) with any shape', () => {
      expect(computeBroadcastShape([[], [3, 4]])).toEqual([3, 4]);
      expect(computeBroadcastShape([[3, 4], []])).toEqual([3, 4]);
    });

    it('broadcasts same shapes', () => {
      expect(
        computeBroadcastShape([
          [3, 4],
          [3, 4],
        ])
      ).toEqual([3, 4]);
    });

    it('broadcasts with trailing dimension match', () => {
      expect(computeBroadcastShape([[5, 4], [4]])).toEqual([5, 4]);
      expect(computeBroadcastShape([[4], [5, 4]])).toEqual([5, 4]);
    });

    it('broadcasts with size 1 dimensions', () => {
      expect(
        computeBroadcastShape([
          [3, 1],
          [1, 4],
        ])
      ).toEqual([3, 4]);
      expect(
        computeBroadcastShape([
          [1, 4],
          [3, 1],
        ])
      ).toEqual([3, 4]);
      expect(
        computeBroadcastShape([
          [8, 1, 6, 1],
          [7, 1, 5],
        ])
      ).toEqual([8, 7, 6, 5]);
    });

    it('broadcasts different number of dimensions', () => {
      expect(computeBroadcastShape([[3, 4, 5], [5]])).toEqual([3, 4, 5]);
      expect(computeBroadcastShape([[5], [3, 4, 5]])).toEqual([3, 4, 5]);
      expect(
        computeBroadcastShape([
          [15, 3, 5],
          [3, 5],
        ])
      ).toEqual([15, 3, 5]);
    });

    it('broadcasts multiple arrays', () => {
      expect(computeBroadcastShape([[6, 7], [5, 6, 1], [7], [5, 1, 7]])).toEqual([5, 6, 7]);
    });

    it('returns null for incompatible shapes', () => {
      expect(computeBroadcastShape([[3], [4]])).toBeNull();
      expect(
        computeBroadcastShape([
          [3, 4],
          [3, 5],
        ])
      ).toBeNull();
      expect(
        computeBroadcastShape([
          [2, 1],
          [8, 4, 3],
        ])
      ).toBeNull();
    });

    it('handles edge case: size 0 dimensions', () => {
      // Size 0 must match with 0 or 1
      expect(computeBroadcastShape([[8, 1, 1, 6, 1], [0]])).toEqual([8, 1, 1, 6, 0]);
      expect(
        computeBroadcastShape([
          [8, 0, 1, 6, 1],
          [6, 5],
        ])
      ).toEqual([8, 0, 1, 6, 5]);
    });
  });

  describe('areBroadcastable', () => {
    it('returns true for compatible shapes', () => {
      expect(areBroadcastable([3, 4], [4])).toBe(true);
      expect(areBroadcastable([3, 1], [1, 4])).toBe(true);
      expect(areBroadcastable([5, 4], [1])).toBe(true);
    });

    it('returns false for incompatible shapes', () => {
      expect(areBroadcastable([3], [4])).toBe(false);
      expect(areBroadcastable([3, 4], [3, 5])).toBe(false);
    });

    it('returns true for same shapes', () => {
      expect(areBroadcastable([3, 4], [3, 4])).toBe(true);
    });
  });

  describe('broadcastArrays', () => {
    it('returns empty array for no inputs', () => {
      const result = broadcastArrays([]);
      expect(result).toEqual([]);
    });

    it('returns same array for single input', () => {
      const a = array([1, 2, 3]);
      const result = broadcastArrays([a.storage]);
      expect(result).toEqual([a.storage]);
    });

    it('broadcasts two compatible arrays', () => {
      const a = ones([3, 4]);
      const b = array([1, 2, 3, 4]);

      const result = broadcastArrays([a.storage, b.storage]);

      expect(result).toHaveLength(2);
      expect(Array.from(result[0]!.shape)).toEqual([3, 4]);
      expect(Array.from(result[1]!.shape)).toEqual([3, 4]);
    });

    it('broadcasts arrays with size-1 dimensions', () => {
      const a = array([[1], [2], [3]]); // shape (3, 1)
      const b = array([[10, 20, 30, 40]]); // shape (1, 4)

      const result = broadcastArrays([a.storage, b.storage]);

      expect(result).toHaveLength(2);
      expect(Array.from(result[0]!.shape)).toEqual([3, 4]);
      expect(Array.from(result[1]!.shape)).toEqual([3, 4]);
    });

    it('broadcasts multiple arrays', () => {
      const a = array([[[1]]]);
      const b = array([[1, 2]]);
      const c = array([[[1], [2], [3]]]);

      const result = broadcastArrays([a.storage, b.storage, c.storage]);

      expect(result).toHaveLength(3);
      // All should be broadcast to (1, 3, 2)
      expect(Array.from(result[0]!.shape)).toEqual([1, 3, 2]);
      expect(Array.from(result[1]!.shape)).toEqual([1, 3, 2]);
      expect(Array.from(result[2]!.shape)).toEqual([1, 3, 2]);
    });

    it('throws error for incompatible shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3, 4]);

      expect(() => broadcastArrays([a.storage, b.storage])).toThrow(
        /operands could not be broadcast together/
      );
    });

    it('throws descriptive error message on broadcast failure', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2, 3]);

      expect(() => broadcastArrays([a.storage, b.storage])).toThrow(
        /operands could not be broadcast together/
      );
    });
  });

  describe('broadcastErrorMessage', () => {
    it('generates error message without operation', () => {
      const shapes = [
        [3, 4],
        [5, 6],
      ];
      const message = broadcastErrorMessage(shapes);

      expect(message).toBe('operands could not be broadcast together with shapes (3,4) (5,6)');
    });

    it('generates error message with operation', () => {
      const shapes = [
        [3, 4],
        [5, 6],
      ];
      const message = broadcastErrorMessage(shapes, 'add');

      expect(message).toBe(
        'operands could not be broadcast together for add with shapes (3,4) (5,6)'
      );
    });

    it('handles single shape', () => {
      const shapes = [[2, 3]];
      const message = broadcastErrorMessage(shapes);

      expect(message).toBe('operands could not be broadcast together with shapes (2,3)');
    });

    it('handles empty shapes (scalars)', () => {
      const shapes = [[], [3, 4]];
      const message = broadcastErrorMessage(shapes, 'multiply');

      expect(message).toBe(
        'operands could not be broadcast together for multiply with shapes () (3,4)'
      );
    });

    it('handles multiple shapes', () => {
      const shapes = [[3, 4], [4], [5, 1]];
      const message = broadcastErrorMessage(shapes, 'subtract');

      expect(message).toBe(
        'operands could not be broadcast together for subtract with shapes (3,4) (4) (5,1)'
      );
    });
  });

  describe('broadcast_shapes', () => {
    it('returns shape for single array', () => {
      expect(broadcast_shapes([3, 4])).toEqual([3, 4]);
    });

    it('broadcasts two compatible shapes', () => {
      expect(broadcast_shapes([3, 4], [4])).toEqual([3, 4]);
      expect(broadcast_shapes([4], [3, 4])).toEqual([3, 4]);
    });

    it('broadcasts with size 1 dimensions', () => {
      expect(broadcast_shapes([3, 1], [1, 4])).toEqual([3, 4]);
      expect(broadcast_shapes([8, 1, 6, 1], [7, 1, 5])).toEqual([8, 7, 6, 5]);
    });

    it('broadcasts multiple shapes', () => {
      expect(broadcast_shapes([6, 7], [5, 6, 1], [7], [5, 1, 7])).toEqual([5, 6, 7]);
    });

    it('broadcasts scalar (empty shape) with any shape', () => {
      expect(broadcast_shapes([], [3, 4])).toEqual([3, 4]);
      expect(broadcast_shapes([3, 4], [])).toEqual([3, 4]);
    });

    it('throws error for incompatible shapes', () => {
      expect(() => broadcast_shapes([3], [4])).toThrow(/Cannot broadcast shapes/);
      expect(() => broadcast_shapes([3, 4], [3, 5])).toThrow(/Cannot broadcast shapes/);
      expect(() => broadcast_shapes([2, 1], [8, 4, 3])).toThrow(/Cannot broadcast shapes/);
    });

    it('throws descriptive error message', () => {
      expect(() => broadcast_shapes([3, 4], [5, 6])).toThrow(
        /Cannot broadcast shapes: dimensions \d+ and \d+ are incompatible/
      );
    });

    it('handles edge case: size 0 dimensions', () => {
      expect(broadcast_shapes([8, 1, 1, 6, 1], [0])).toEqual([8, 1, 1, 6, 0]);
      expect(broadcast_shapes([8, 0, 1, 6, 1], [6, 5])).toEqual([8, 0, 1, 6, 5]);
    });

    it('works with many shapes at once', () => {
      expect(broadcast_shapes([256, 256, 3], [3], [1])).toEqual([256, 256, 3]);
    });
  });
});

describe('Broadcasting in arithmetic operations', () => {
  describe('add with broadcasting', () => {
    it('broadcasts (3, 4) + (4,)', () => {
      const a = ones([3, 4]);
      const b = array([1, 2, 3, 4]);

      const result = a.add(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
      ]);
    });

    it('broadcasts (4,) + (3, 4)', () => {
      const a = array([1, 2, 3, 4]);
      const b = ones([3, 4]);

      const result = a.add(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
      ]);
    });

    it('broadcasts (3, 1) + (1, 4)', () => {
      const a = array([[1], [2], [3]]);
      const b = array([[10, 20, 30, 40]]);

      const result = a.add(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [11, 21, 31, 41],
        [12, 22, 32, 42],
        [13, 23, 33, 43],
      ]);
    });

    it('broadcasts same shapes (no actual broadcasting)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [10, 20],
        [30, 40],
      ]);

      const result = a.add(b);

      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [11, 22],
        [33, 44],
      ]);
    });

    it('throws error for incompatible shapes', () => {
      const a = array([1, 2, 3]);
      const b = array([1, 2, 3, 4]);

      expect(() => a.add(b)).toThrow(/operands could not be broadcast/);
    });
  });

  describe('subtract with broadcasting', () => {
    it('broadcasts (3, 4) - (4,)', () => {
      const a = ones([3, 4]).multiply(10);
      const b = array([1, 2, 3, 4]);

      const result = a.subtract(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [9, 8, 7, 6],
        [9, 8, 7, 6],
        [9, 8, 7, 6],
      ]);
    });

    it('broadcasts (3, 1) - (1, 4)', () => {
      const a = array([[10], [20], [30]]);
      const b = array([[1, 2, 3, 4]]);

      const result = a.subtract(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [9, 8, 7, 6],
        [19, 18, 17, 16],
        [29, 28, 27, 26],
      ]);
    });
  });

  describe('multiply with broadcasting', () => {
    it('broadcasts (3, 4) * (4,)', () => {
      const a = ones([3, 4]).multiply(2);
      const b = array([1, 2, 3, 4]);

      const result = a.multiply(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [2, 4, 6, 8],
        [2, 4, 6, 8],
        [2, 4, 6, 8],
      ]);
    });

    it('broadcasts (3, 1) * (1, 4)', () => {
      const a = array([[2], [3], [4]]);
      const b = array([[10, 20, 30, 40]]);

      const result = a.multiply(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [20, 40, 60, 80],
        [30, 60, 90, 120],
        [40, 80, 120, 160],
      ]);
    });
  });

  describe('divide with broadcasting', () => {
    it('broadcasts (3, 4) / (4,)', () => {
      const a = array([
        [10, 20, 30, 40],
        [10, 20, 30, 40],
        [10, 20, 30, 40],
      ]);
      const b = array([2, 4, 5, 10]);

      const result = a.divide(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [5, 5, 6, 4],
        [5, 5, 6, 4],
        [5, 5, 6, 4],
      ]);
    });

    it('broadcasts (3, 1) / (1, 4)', () => {
      const a = array([[100], [200], [400]]);
      const b = array([[10, 20, 25, 50]]);

      const result = a.divide(b);

      expect(result.shape).toEqual([3, 4]);
      expect(result.toArray()).toEqual([
        [10, 5, 4, 2],
        [20, 10, 8, 4],
        [40, 20, 16, 8],
      ]);
    });
  });

  describe('complex broadcasting scenarios', () => {
    it('broadcasts 3D with 1D', () => {
      const a = zeros([2, 3, 4]).add(1);
      const b = array([1, 2, 3, 4]);

      const result = a.add(b);

      expect(result.shape).toEqual([2, 3, 4]);
      // Each 3x4 slice should have the same pattern
      const slice = result.toArray()[0] as number[][];
      expect(slice).toEqual([
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
      ]);
    });

    it('broadcasts with multiple size-1 dimensions', () => {
      const a = array([[[1]], [[2]], [[3]]]);
      const b = array([[10, 20]]);

      const result = a.multiply(b);

      expect(result.shape).toEqual([3, 1, 2]);
      expect(result.toArray()).toEqual([[[10, 20]], [[20, 40]], [[30, 60]]]);
    });
  });
});
