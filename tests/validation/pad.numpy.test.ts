/**
 * NumPy validation tests for np.pad — verifies per-axis pad_width and
 * per-axis constant_values support.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { arange, pad } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: pad', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('uniform scalar pad_width and value (regression)', () => {
    const a = arange(6).reshape(2, 3);
    const result = pad(a, 1, 'constant', 7);
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
result = np.pad(a, 1, mode='constant', constant_values=7)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('per-axis pad_width as nested pairs', () => {
    const a = arange(6).reshape(2, 3);
    const result = pad(
      a,
      [
        [1, 2],
        [3, 0],
      ],
      'constant',
      0,
    );
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
result = np.pad(a, [[1, 2], [3, 0]], mode='constant')
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('per-axis constant_values as pair of pairs', () => {
    const a = arange(6).reshape(2, 3);
    const result = pad(a, 1, 'constant', [
      [10, 20],
      [30, 40],
    ]);
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
result = np.pad(a, 1, mode='constant', constant_values=[[10, 20], [30, 40]])
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('asymmetric pair broadcast across axes', () => {
    const a = arange(6).reshape(2, 3);
    const result = pad(a, [1, 2], 'constant', [3, 9]);
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
result = np.pad(a, [1, 2], mode='constant', constant_values=[3, 9])
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('per-axis scalars as 1-D list with length=ndim', () => {
    const a = arange(6).reshape(2, 3);
    const result = pad(a, [1, 2], 'constant', 0);
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
# In NumPy, [1, 2] for 2-D array is interpreted as [before, after] applied to all axes.
result = np.pad(a, [1, 2], mode='constant')
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('mixed scalars and pairs for constant_values', () => {
    const a = arange(6).reshape(2, 3);
    const result = pad(a, 1, 'constant', [5, [7, 8]]);
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
# axis 0: before=after=5; axis 1: before=7, after=8
result = np.pad(a, 1, mode='constant', constant_values=[[5, 5], [7, 8]])
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });
});
