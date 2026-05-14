/**
 * Nested view mutation semantics.
 *
 * Slicing returns a view that shares the parent's data buffer. The `_base`
 * chain in NDArrayCore (and the `up()` recursion in src/full/ndarray.ts:18-22)
 * flattens these so that a slice-of-a-slice points back at the original root,
 * not at the intermediate view. This test pins the observable contract:
 * writes through the deepest view must show up at the root.
 */

import { describe, expect, it } from 'vitest';
import { arange, array, zeros } from '../../src';

describe('nested view mutation', () => {
  describe('1D slice chain', () => {
    it('mutation through a 3-level slice is visible at the root', () => {
      const root = arange(20); // [0,1,...,19]
      const a = root.slice('2:18'); // [2..17]
      const b = a.slice('1:15'); // [3..16]
      const c = b.slice('2:10'); // [5..12]
      c.set([0], 999);
      // c[0] === root[5]
      expect(root.get([5])).toBe(999);
      expect(a.get([3])).toBe(999); // a starts at root[2], so root[5] is a[3]
      expect(b.get([2])).toBe(999); // b starts at root[3]
      expect(c.get([0])).toBe(999);
    });

    it('mutation through a sliced view of an axis-stepped view', () => {
      const root = arange(20);
      const even = root.slice('::2'); // [0,2,4,...,18]
      const tail = even.slice('3:7'); // [6,8,10,12]
      tail.set([2], -1); // mutates root[10]
      expect(root.get([10])).toBe(-1);
      expect(even.get([5])).toBe(-1);
      expect(tail.get([2])).toBe(-1);
    });
  });

  describe('2D slice chain', () => {
    it('row-then-column slice writes reach root', () => {
      const root = zeros([5, 6]);
      const sub = root.slice('1:4', '1:5'); // 3×4
      const inner = sub.slice('1:3', '1:3'); // 2×2 starting at root[2,2]
      inner.set([0, 0], 7);
      expect(root.get([2, 2])).toBe(7);
      expect(sub.get([1, 1])).toBe(7);
      expect(inner.get([0, 0])).toBe(7);
    });

    it('reversed-stride view chain still mutates root', () => {
      const root = arange(12).reshape(3, 4);
      const flipped = root.slice('::-1', '::-1'); // reverses both axes
      const inner = flipped.slice('0:2', '0:2');
      inner.set([0, 0], 42);
      // flipped[0,0] is root[2,3] (= 11 originally)
      expect(root.get([2, 3])).toBe(42);
    });
  });

  describe('base attribute always points to the ultimate root', () => {
    it('slice-of-slice.base === root, not the intermediate view', () => {
      const root = arange(20);
      const a = root.slice('2:18');
      const b = a.slice('1:15');
      const c = b.slice('2:10');
      expect(a.base).toBe(root);
      expect(b.base).toBe(root);
      expect(c.base).toBe(root);
    });

    it('root.base is null', () => {
      const root = arange(8);
      expect(root.base).toBeNull();
    });

    it('flatten survives mixed indexing styles', () => {
      const root = zeros([4, 5]);
      const r = root.slice('1:4'); // first-axis slice
      const c = r.slice(':', '1:4'); // second-axis slice on a view
      const inner = c.slice('0:2', '0:2');
      expect(r.base).toBe(root);
      expect(c.base).toBe(root);
      expect(inner.base).toBe(root);
      inner.set([1, 1], 5);
      // inner[1,1] = c[1,1] = r[1,2] = root[2,2]
      expect(root.get([2, 2])).toBe(5);
    });
  });

  describe('shared-buffer invariants', () => {
    it('all slices in a chain share the same TypedArray buffer', () => {
      const root = arange(12).reshape(3, 4);
      const a = root.slice('0:3', '1:4');
      const b = a.slice('1:3', '0:2');
      expect(a.data.buffer).toBe(root.data.buffer);
      expect(b.data.buffer).toBe(root.data.buffer);
    });

    it('a copy() breaks the chain — mutating the copy does not affect the root', () => {
      const root = arange(20);
      const a = root.slice('2:18');
      const detached = a.copy();
      detached.set([0], 999);
      expect(root.get([2])).toBe(2); // unchanged
      expect(detached.base).toBeNull();
    });
  });

  describe('reverse direction: mutating root is visible through views', () => {
    it('writes to root propagate to all open views', () => {
      const root = array([10, 20, 30, 40, 50]);
      const a = root.slice('1:5');
      const b = a.slice('0:3');
      root.set([2], 99);
      expect(a.get([1])).toBe(99); // a[1] === root[2]
      expect(b.get([1])).toBe(99); // b[1] === a[1] === root[2]
    });
  });
});
