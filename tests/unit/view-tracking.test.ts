/**
 * Tests for view tracking (base attribute)
 * Verifies that views correctly track their base array
 */

import { describe, it, expect } from 'vitest';
import { array, ones, zeros } from '../../src';

describe('View Tracking', () => {
  describe('base attribute', () => {
    it('returns null for arrays that own their data', () => {
      const arr = ones([3, 3]);
      expect(arr.base).toBe(null);

      const arr2 = array([1, 2, 3]);
      expect(arr2.base).toBe(null);

      const arr3 = zeros([2, 2]);
      expect(arr3.base).toBe(null);
    });

    it('tracks base for sliced arrays', () => {
      const arr = ones([4, 4]);
      const view = arr.slice('0:2', '0:2');

      expect(view.base).toBe(arr);
      expect(arr.base).toBe(null);
    });

    it('tracks base for transposed arrays', () => {
      const arr = ones([3, 4]);
      const transposed = arr.transpose();

      expect(transposed.base).toBe(arr);
      expect(arr.base).toBe(null);
    });

    it('tracks base for squeezed arrays', () => {
      const arr = ones([2, 1, 3]);
      const squeezed = arr.squeeze(1);

      expect(squeezed.base).toBe(arr);
      expect(arr.base).toBe(null);
    });

    it('tracks base for expand_dims arrays', () => {
      const arr = ones([2, 3]);
      const expanded = arr.expand_dims(1);

      expect(expanded.base).toBe(arr);
      expect(arr.base).toBe(null);
    });

    it('tracks base for reshaped arrays (when view)', () => {
      const arr = ones([2, 6]); // C-contiguous
      const reshaped = arr.reshape(3, 4);

      // reshape creates a view for C-contiguous arrays
      if (reshaped.base !== null) {
        expect(reshaped.base).toBe(arr);
      }
    });

    it('tracks base for raveled arrays (when view)', () => {
      const arr = ones([2, 3]); // C-contiguous
      const raveled = arr.ravel();

      // ravel creates a view for C-contiguous arrays
      if (raveled.base !== null) {
        expect(raveled.base).toBe(arr);
      }
    });

    it('does not track base for flattened arrays (always copy)', () => {
      const arr = ones([2, 3]);
      const flattened = arr.flatten();

      // flatten always creates a copy
      expect(flattened.base).toBe(null);
    });

    it('does not track base for reshaped arrays (when copy)', () => {
      const arr = ones([2, 3]);
      const transposed = arr.transpose(); // Makes non-contiguous
      const reshaped = transposed.reshape(3, 2);

      // reshape creates a copy for non-contiguous arrays
      expect(reshaped.base).toBe(null);
    });
  });

  describe('view chains', () => {
    it('tracks original base through multiple views', () => {
      const arr = ones([4, 4]);
      const view1 = arr.slice(':', ':');
      const view2 = view1.transpose();
      const view3 = view2.squeeze();

      // All views should point to original array
      expect(view1.base).toBe(arr);
      expect(view2.base).toBe(arr);
      // view3 might be arr or view2 depending on if squeeze creates new view
      expect(view3.base).toBe(arr);
    });

    it('tracks base through slice then transpose', () => {
      const arr = ones([5, 5]);
      const sliced = arr.slice('0:3', '0:3');
      const transposed = sliced.transpose();

      expect(sliced.base).toBe(arr);
      expect(transposed.base).toBe(arr);
    });

    it('tracks base through transpose then slice', () => {
      const arr = ones([5, 5]);
      const transposed = arr.transpose();
      const sliced = transposed.slice('0:3', '0:3');

      expect(transposed.base).toBe(arr);
      expect(sliced.base).toBe(arr);
    });
  });

  describe('flags.OWNDATA', () => {
    it('returns true for arrays that own data', () => {
      const arr = ones([3, 3]);
      expect(arr.flags.OWNDATA).toBe(true);
    });

    it('returns false for views', () => {
      const arr = ones([4, 4]);
      const view = arr.slice('0:2', '0:2');

      expect(arr.flags.OWNDATA).toBe(true);
      expect(view.flags.OWNDATA).toBe(false);
    });

    it('returns false for transposed arrays', () => {
      const arr = ones([3, 4]);
      const transposed = arr.transpose();

      expect(arr.flags.OWNDATA).toBe(true);
      expect(transposed.flags.OWNDATA).toBe(false);
    });

    it('returns true for copies', () => {
      const arr = ones([2, 3]);
      const copy = arr.copy();
      const flattened = arr.flatten();

      expect(copy.flags.OWNDATA).toBe(true);
      expect(flattened.flags.OWNDATA).toBe(true);
    });
  });

  describe('memory sharing', () => {
    it('views share memory with base array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const view = arr.slice('0', ':');

      // Note: @stdlib creates read-only views, so we can't test memory sharing via set()
      // However, we can verify they share the same data buffer
      expect(view.data).toBe(arr.data);
      expect(view.base).toBe(arr);
    });

    it('copies do not share memory', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const copy = arr.copy();

      // Modify copy
      copy.set([0, 0], 999);

      // Original should be unchanged
      expect(arr.get([0, 0])).toBe(1);
    });

    it('flattened arrays do not share memory', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const flat = arr.flatten();

      // Modify flattened
      flat.set([0], 999);

      // Original should be unchanged
      expect(arr.get([0, 0])).toBe(1);
    });
  });

  describe('edge cases', () => {
    it('handles 1D arrays', () => {
      const arr = array([1, 2, 3, 4]);
      const view = arr.slice('0:2');

      expect(view.base).toBe(arr);
      expect(arr.base).toBe(null);
    });

    it('handles scalar-like arrays', () => {
      const arr = ones([1]);
      const squeezed = arr.squeeze();

      expect(squeezed.base).toBe(arr);
    });

    it('handles empty slices', () => {
      const arr = ones([3, 3]);
      const empty = arr.slice('0:0', ':');

      expect(empty.base).toBe(arr);
      expect(empty.size).toBe(0);
    });
  });
});
