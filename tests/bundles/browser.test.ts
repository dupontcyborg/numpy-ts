/**
 * Smoke test for Browser ESM bundle
 * Tests the actual distributed bundle from dist/ in a real browser
 */

import { describe, test, expect, beforeAll } from 'vitest';

let np: any;

describe('Browser ESM Bundle Smoke Test', () => {
  beforeAll(async () => {
    // Load the ESM bundle via dynamic import
    np = await import('/dist/numpy-ts.browser.js');
  });

  test('should export main functions', async () => {
    expect(typeof np).toBe('object');
    expect(typeof np.array).toBe('function');
    expect(typeof np.zeros).toBe('function');
    expect(typeof np.ones).toBe('function');
    expect(typeof np.arange).toBe('function');
  });

  test('should create arrays', () => {
    const arr = np.array([1, 2, 3, 4]);
    expect(arr.shape).toEqual([4]);
    expect(arr.toArray()).toEqual([1, 2, 3, 4]);
  });

  test('should perform basic math', () => {
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const result = a.add(b);
    expect(result.toArray()).toEqual([5, 7, 9]);
  });

  test('should handle matrix operations', () => {
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    const result = a.matmul(b);
    expect(result.shape).toEqual([2, 2]);
    expect(result.toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  test('should create zeros and ones', () => {
    const z = np.zeros([2, 3]);
    expect(z.shape).toEqual([2, 3]);
    expect(z.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const o = np.ones([3, 2]);
    expect(o.shape).toEqual([3, 2]);
    expect(o.toArray()).toEqual([
      [1, 1],
      [1, 1],
      [1, 1],
    ]);
  });

  // ── File IO should throw in the browser ─────────────────────────────

  test('loadNpy should throw a helpful error in the browser', async () => {
    expect(typeof np.loadNpy).toBe('function');
    try {
      await np.loadNpy('test.npy');
      expect.unreachable('loadNpy should have thrown');
    } catch (e: any) {
      expect(e.message).toContain('requires Node.js, Bun, or Deno');
      expect(e.message).toContain('parseNpy');
    }
  });

  test('loadNpySync should throw a helpful error in the browser', () => {
    expect(typeof np.loadNpySync).toBe('function');
    expect(() => {
      np.loadNpySync('test.npy');
    }).toThrow('requires Node.js, Bun, or Deno');
  });

  test('saveNpy should throw a helpful error in the browser', async () => {
    const arr = np.array([1, 2, 3]);
    try {
      await np.saveNpy('test.npy', arr);
      expect.unreachable('saveNpy should have thrown');
    } catch (e: any) {
      expect(e.message).toContain('requires Node.js, Bun, or Deno');
    }
  });

  test('loadtxt should throw a helpful error in the browser', async () => {
    try {
      await np.loadtxt('data.csv');
      expect.unreachable('loadtxt should have thrown');
    } catch (e: any) {
      expect(e.message).toContain('requires Node.js, Bun, or Deno');
    }
  });
});
