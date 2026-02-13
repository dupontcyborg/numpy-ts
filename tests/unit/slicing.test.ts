/**
 * Unit tests for slicing utilities
 */

import { describe, it, expect } from 'vitest';
import {
  parseSlice,
  normalizeSlice,
  computeSliceLength,
  parseSlices,
} from '../../src/common/slicing';

describe('Slicing utilities', () => {
  describe('parseSlice', () => {
    describe('single index', () => {
      it('parses positive index', () => {
        const result = parseSlice('5');
        expect(result).toEqual({
          start: 5,
          stop: null,
          step: 1,
          isIndex: true,
        });
      });

      it('parses negative index', () => {
        const result = parseSlice('-1');
        expect(result).toEqual({
          start: -1,
          stop: null,
          step: 1,
          isIndex: true,
        });
      });

      it('parses zero index', () => {
        const result = parseSlice('0');
        expect(result).toEqual({
          start: 0,
          stop: null,
          step: 1,
          isIndex: true,
        });
      });

      it('throws on invalid index', () => {
        expect(() => parseSlice('abc')).toThrow(/Invalid slice index/);
        expect(() => parseSlice('1.5')).toThrow(/Invalid slice index/);
      });
    });

    describe('full slice notation', () => {
      it('parses start:stop', () => {
        const result = parseSlice('0:5');
        expect(result).toEqual({
          start: 0,
          stop: 5,
          step: 1,
          isIndex: false,
        });
      });

      it('parses negative indices', () => {
        const result = parseSlice('-5:-2');
        expect(result).toEqual({
          start: -5,
          stop: -2,
          step: 1,
          isIndex: false,
        });
      });

      it('parses mixed signs', () => {
        const result = parseSlice('-5:10');
        expect(result).toEqual({
          start: -5,
          stop: 10,
          step: 1,
          isIndex: false,
        });
      });
    });

    describe('slice with step', () => {
      it('parses start:stop:step', () => {
        const result = parseSlice('0:10:2');
        expect(result).toEqual({
          start: 0,
          stop: 10,
          step: 2,
          isIndex: false,
        });
      });

      it('parses ::step', () => {
        const result = parseSlice('::2');
        expect(result).toEqual({
          start: null,
          stop: null,
          step: 2,
          isIndex: false,
        });
      });

      it('parses negative step', () => {
        const result = parseSlice('::-1');
        expect(result).toEqual({
          start: null,
          stop: null,
          step: -1,
          isIndex: false,
        });
      });

      it('parses start::step', () => {
        const result = parseSlice('5::2');
        expect(result).toEqual({
          start: 5,
          stop: null,
          step: 2,
          isIndex: false,
        });
      });

      it('throws on zero step', () => {
        expect(() => parseSlice('::0')).toThrow(/step cannot be zero/);
      });
    });

    describe('partial slices', () => {
      it('parses ":"', () => {
        const result = parseSlice(':');
        expect(result).toEqual({
          start: null,
          stop: null,
          step: 1,
          isIndex: false,
        });
      });

      it('parses "5:"', () => {
        const result = parseSlice('5:');
        expect(result).toEqual({
          start: 5,
          stop: null,
          step: 1,
          isIndex: false,
        });
      });

      it('parses ":10"', () => {
        const result = parseSlice(':10');
        expect(result).toEqual({
          start: null,
          stop: 10,
          step: 1,
          isIndex: false,
        });
      });

      it('parses ":-5"', () => {
        const result = parseSlice(':-5');
        expect(result).toEqual({
          start: null,
          stop: -5,
          step: 1,
          isIndex: false,
        });
      });

      it('parses "-5:"', () => {
        const result = parseSlice('-5:');
        expect(result).toEqual({
          start: -5,
          stop: null,
          step: 1,
          isIndex: false,
        });
      });
    });

    describe('error cases', () => {
      it('throws on too many colons', () => {
        expect(() => parseSlice('1:2:3:4')).toThrow(/too many colons/);
      });

      it('throws on invalid start', () => {
        expect(() => parseSlice('abc:5')).toThrow(/Invalid start index/);
      });

      it('throws on invalid stop', () => {
        expect(() => parseSlice('0:abc')).toThrow(/Invalid stop index/);
      });

      it('throws on invalid step', () => {
        expect(() => parseSlice('0:10:abc')).toThrow(/Invalid step/);
      });
    });
  });

  describe('normalizeSlice', () => {
    describe('single index normalization', () => {
      it('normalizes positive index', () => {
        const spec = parseSlice('5');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 5,
          stop: 6,
          step: 1,
          isIndex: true,
        });
      });

      it('normalizes negative index', () => {
        const spec = parseSlice('-1');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 9,
          stop: 10,
          step: 1,
          isIndex: true,
        });
      });

      it('normalizes -2', () => {
        const spec = parseSlice('-2');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 8,
          stop: 9,
          step: 1,
          isIndex: true,
        });
      });

      it('throws on out of bounds positive index', () => {
        const spec = parseSlice('10');
        expect(() => normalizeSlice(spec, 10)).toThrow(/out of bounds/);
      });

      it('throws on out of bounds negative index', () => {
        const spec = parseSlice('-11');
        expect(() => normalizeSlice(spec, 10)).toThrow(/out of bounds/);
      });
    });

    describe('forward slice normalization (positive step)', () => {
      it('normalizes explicit start:stop', () => {
        const spec = parseSlice('2:7');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 2,
          stop: 7,
          step: 1,
          isIndex: false,
        });
      });

      it('normalizes ":" to full range', () => {
        const spec = parseSlice(':');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 0,
          stop: 10,
          step: 1,
          isIndex: false,
        });
      });

      it('normalizes "5:" to start from 5', () => {
        const spec = parseSlice('5:');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 5,
          stop: 10,
          step: 1,
          isIndex: false,
        });
      });

      it('normalizes ":5" to stop at 5', () => {
        const spec = parseSlice(':5');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 0,
          stop: 5,
          step: 1,
          isIndex: false,
        });
      });

      it('handles negative start', () => {
        const spec = parseSlice('-5:');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 5,
          stop: 10,
          step: 1,
          isIndex: false,
        });
      });

      it('handles negative stop', () => {
        const spec = parseSlice(':-2');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 0,
          stop: 8,
          step: 1,
          isIndex: false,
        });
      });

      it('clamps start to valid range', () => {
        const spec = parseSlice('-20:5');
        const result = normalizeSlice(spec, 10);
        expect(result.start).toBe(0);
      });

      it('clamps stop to valid range', () => {
        const spec = parseSlice('0:20');
        const result = normalizeSlice(spec, 10);
        expect(result.stop).toBe(10);
      });
    });

    describe('backward slice normalization (negative step)', () => {
      it('normalizes ::-1 (reverse)', () => {
        const spec = parseSlice('::-1');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 9,
          stop: -1,
          step: -1,
          isIndex: false,
        });
      });

      it('normalizes 7:2:-1', () => {
        const spec = parseSlice('7:2:-1');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 7,
          stop: 2,
          step: -1,
          isIndex: false,
        });
      });

      it('handles negative indices with negative step', () => {
        const spec = parseSlice('-3:-8:-1');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 7,
          stop: 2,
          step: -1,
          isIndex: false,
        });
      });
    });

    describe('with step', () => {
      it('normalizes ::2', () => {
        const spec = parseSlice('::2');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 0,
          stop: 10,
          step: 2,
          isIndex: false,
        });
      });

      it('normalizes 1:8:2', () => {
        const spec = parseSlice('1:8:2');
        const result = normalizeSlice(spec, 10);
        expect(result).toEqual({
          start: 1,
          stop: 8,
          step: 2,
          isIndex: false,
        });
      });
    });
  });

  describe('computeSliceLength', () => {
    it('computes length for forward slice', () => {
      expect(computeSliceLength(0, 10, 1)).toBe(10);
      expect(computeSliceLength(2, 7, 1)).toBe(5);
      expect(computeSliceLength(0, 10, 2)).toBe(5);
      expect(computeSliceLength(1, 10, 2)).toBe(5);
      expect(computeSliceLength(0, 10, 3)).toBe(4);
    });

    it('computes length for backward slice', () => {
      expect(computeSliceLength(9, -1, -1)).toBe(10);
      expect(computeSliceLength(7, 2, -1)).toBe(5);
      expect(computeSliceLength(9, -1, -2)).toBe(5);
    });

    it('returns 0 for empty slices', () => {
      expect(computeSliceLength(5, 5, 1)).toBe(0);
      expect(computeSliceLength(5, 3, 1)).toBe(0);
      expect(computeSliceLength(3, 5, -1)).toBe(0);
    });
  });

  describe('parseSlices', () => {
    it('parses multiple slice strings', () => {
      const result = parseSlices(['0:5', ':', '::2']);
      expect(result).toHaveLength(3);
      expect(result[0]).toEqual({
        start: 0,
        stop: 5,
        step: 1,
        isIndex: false,
      });
      expect(result[1]).toEqual({
        start: null,
        stop: null,
        step: 1,
        isIndex: false,
      });
      expect(result[2]).toEqual({
        start: null,
        stop: null,
        step: 2,
        isIndex: false,
      });
    });

    it('handles empty array', () => {
      const result = parseSlices([]);
      expect(result).toEqual([]);
    });

    it('handles single slice', () => {
      const result = parseSlices([':']);
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        start: null,
        stop: null,
        step: 1,
        isIndex: false,
      });
    });
  });
});
