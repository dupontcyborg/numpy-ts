import { afterEach, describe, expect, it } from 'vitest';
import * as np from '../../src';

describe('narrow float rounding without Math.f16round', () => {
  const original = (Math as { f16round?: unknown }).f16round;
  afterEach(() => {
    Object.defineProperty(Math, 'f16round', {
      value: original,
      configurable: true,
      writable: true,
    });
  });

  it('unwrap on float16 falls back to float32 rounding', () => {
    Object.defineProperty(Math, 'f16round', {
      value: undefined,
      configurable: true,
      writable: true,
    });
    const r = np.unwrap(np.array([0, 1, 2, 3, -3], 'float16'));
    expect(r.dtype).toBe('float16');
    expect(Array.from(r.toArray() as number[]).map((v) => Math.round(v * 100) / 100)).toEqual([
      0, 1, 2, 3, 3.28,
    ]);
  });

  it('histogram_bin_edges on float16 falls back to float32 rounding', () => {
    Object.defineProperty(Math, 'f16round', {
      value: undefined,
      configurable: true,
      writable: true,
    });
    const e = np.histogram_bin_edges(np.array([0, 1], 'float16'), 4);
    expect(e.dtype).toBe('float16');
    expect(e.toArray()).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });
});

describe('histogram family edge cases', () => {
  it('histogram2d rejects explicit edges that are not 1-D', () => {
    expect(() =>
      np.histogram2d(np.array([1, 2]), np.array([1, 2]), [
        np.array([
          [0, 1],
          [2, 3],
        ]),
        np.array([0, 3]),
      ]),
    ).toThrow(/1d/);
  });

  it('histogram uses the real part of complex bin edges', () => {
    const bins = np.array([0, 2, 4]).astype('complex128');
    const [hist, edges] = np.histogram(np.array([1, 3, 3]), bins);
    expect(hist.toArray()).toEqual([1, 2]);
    expect(edges.toArray()).toEqual([0, 2, 4]);
  });

  it('histogram of complex data uses the real part', () => {
    const [hist] = np.histogram(np.array([1, 3, 3]).astype('complex128'), 2, [0, 4]);
    expect(hist.toArray()).toEqual([1, 2]);
  });
});
