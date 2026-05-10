/**
 * NumPy validation tests confirming bincount/histogram/histogram2d/histogramdd/digitize
 * accept array-like inputs (plain number[]) without manual array() wrapping.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { bincount, digitize, histogram, histogram2d, histogramdd } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: histogram-family ArrayLike inputs', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('bincount accepts plain number[]', () => {
    const result = bincount([0, 1, 1, 2, 2, 2, 3], [0.5, 1, 1, 2, 2, 2, 3]);
    const np = runNumPy(`
result = np.bincount(np.array([0, 1, 1, 2, 2, 2, 3]),
                     weights=np.array([0.5, 1, 1, 2, 2, 2, 3]))
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('digitize accepts plain number[] for x and bins', () => {
    const result = digitize([0.2, 6.4, 3.0, 1.6], [0.0, 1.0, 2.5, 4.0, 10.0]);
    const np = runNumPy(`
result = np.digitize(np.array([0.2, 6.4, 3.0, 1.6]), np.array([0.0, 1.0, 2.5, 4.0, 10.0]))
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('histogram accepts plain number[]', () => {
    const [hist] = histogram([1, 2, 1, 2, 3, 4, 5, 5, 5], 5);
    const np = runNumPy(`
hist, _ = np.histogram(np.array([1, 2, 1, 2, 3, 4, 5, 5, 5]), bins=5)
result = hist
`);
    expect(Array.from(hist.shape)).toEqual(np.shape);
    expect(arraysClose(hist.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('histogram bins as plain number[]', () => {
    const [hist] = histogram([0.1, 0.5, 0.9, 1.2, 2.7], [0, 1, 2, 3]);
    const np = runNumPy(`
hist, _ = np.histogram(np.array([0.1, 0.5, 0.9, 1.2, 2.7]), bins=[0, 1, 2, 3])
result = hist
`);
    expect(Array.from(hist.shape)).toEqual(np.shape);
    expect(arraysClose(hist.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('histogram2d accepts plain number[]', () => {
    const [h] = histogram2d([1, 2, 3, 4], [1, 1, 4, 4], 2);
    const np = runNumPy(`
h, _, _ = np.histogram2d(np.array([1, 2, 3, 4]), np.array([1, 1, 4, 4]), bins=2)
result = h
`);
    expect(Array.from(h.shape)).toEqual(np.shape);
    expect(arraysClose(h.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('histogram2d bins as [number[], number[]]', () => {
    const [h] = histogram2d(
      [1, 2, 3, 4],
      [1, 1, 4, 4],
      [
        [0, 2, 4],
        [0, 2, 4],
      ],
    );
    const np = runNumPy(`
h, _, _ = np.histogram2d(np.array([1, 2, 3, 4]), np.array([1, 1, 4, 4]),
                         bins=[[0, 2, 4], [0, 2, 4]])
result = h
`);
    expect(Array.from(h.shape)).toEqual(np.shape);
    expect(arraysClose(h.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('histogramdd accepts plain nested array', () => {
    const [h] = histogramdd(
      [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ],
      2,
    );
    const np = runNumPy(`
sample = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
h, _ = np.histogramdd(sample, bins=2)
result = h
`);
    expect(Array.from(h.shape)).toEqual(np.shape);
    expect(arraysClose(h.toArray() as number[], np.value, 1e-10)).toBe(true);
  });
});
