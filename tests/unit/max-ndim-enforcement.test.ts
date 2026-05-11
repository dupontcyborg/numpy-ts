/**
 * MAX_NDIM enforcement at rank-growing entry points.
 *
 * Every public op that can increase an array's rank must funnel through the
 * ArrayStorage constructor, which is the single source of truth for the
 * MAX_NDIM ceiling. If a new op bypasses the constructor in the future, this
 * test surfaces it.
 */

import { describe, expect, it } from 'vitest';
import { array, broadcast_to, expand_dims, ones, reshape, stack, zeros } from '../../src';
import { MAX_NDIM } from '../../src/common/storage';

function rankShape(ndim: number, size = 1): number[] {
  // [1, 1, 1, ..., size] — fills `ndim` dims with the trailing one holding `size`.
  return Array.from({ length: ndim }, (_, i) => (i === ndim - 1 ? size : 1));
}

describe('MAX_NDIM enforcement', () => {
  it('is set to 64 (sanity)', () => {
    expect(MAX_NDIM).toBe(64);
  });

  describe('rank-growing ops reject ndim > MAX_NDIM', () => {
    it('reshape', () => {
      const flat = array(new Array(1).fill(0));
      const target = rankShape(MAX_NDIM + 1, 1);
      expect(() => reshape(flat, target)).toThrow(/maximum supported dimension/);
    });

    it('expand_dims at MAX_NDIM', () => {
      const a = zeros(rankShape(MAX_NDIM, 1));
      expect(() => expand_dims(a, 0)).toThrow(/maximum supported dimension/);
    });

    it('stack of MAX_NDIM-rank arrays', () => {
      const a = zeros(rankShape(MAX_NDIM, 1));
      const b = zeros(rankShape(MAX_NDIM, 1));
      expect(() => stack([a, b], 0)).toThrow(/maximum supported dimension/);
    });

    it('broadcast_to a too-many-dim target shape', () => {
      const a = ones([1]);
      const target = rankShape(MAX_NDIM + 1, 1);
      expect(() => broadcast_to(a, target)).toThrow(/maximum supported dimension/);
    });

    it('zeros / ones factories', () => {
      const target = rankShape(MAX_NDIM + 1, 1);
      expect(() => zeros(target)).toThrow(/maximum supported dimension/);
      expect(() => ones(target)).toThrow(/maximum supported dimension/);
    });
  });

  describe('rank-growing ops accept exactly MAX_NDIM', () => {
    it('reshape to MAX_NDIM dims', () => {
      const flat = zeros([1]);
      const out = reshape(flat, rankShape(MAX_NDIM, 1));
      expect(out.shape).toHaveLength(MAX_NDIM);
    });

    it('expand_dims from MAX_NDIM-1 to MAX_NDIM', () => {
      const a = zeros(rankShape(MAX_NDIM - 1, 1));
      const out = expand_dims(a, 0);
      expect(out.shape).toHaveLength(MAX_NDIM);
    });

    it('stack of MAX_NDIM-1 rank arrays to reach MAX_NDIM', () => {
      const a = zeros(rankShape(MAX_NDIM - 1, 1));
      const b = zeros(rankShape(MAX_NDIM - 1, 1));
      const out = stack([a, b], 0);
      expect(out.shape).toHaveLength(MAX_NDIM);
    });

    it('broadcast_to MAX_NDIM target shape', () => {
      const a = ones([1]);
      const out = broadcast_to(a, rankShape(MAX_NDIM, 1));
      expect(out.shape).toHaveLength(MAX_NDIM);
    });
  });
});
