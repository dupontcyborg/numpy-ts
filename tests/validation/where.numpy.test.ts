/**
 * NumPy validation tests for np.where — verifies scalar / array-like x/y inputs.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import type { NDArray } from '../../src';
import { arange, array, where } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: where', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('accepts scalar x and y', () => {
    const cond = array([true, false, true, false]);
    const result = where(cond, 1, 0) as NDArray;
    const np = runNumPy(`
cond = np.array([True, False, True, False])
result = np.where(cond, 1, 0)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('accepts scalar y with falsy 0', () => {
    // Regression: previous `if (x)` check treated x=0 as undefined (bug).
    const cond = array([true, false, true]);
    const x = array([10, 20, 30]);
    const result = where(cond, x, 0) as NDArray;
    const np = runNumPy(`
cond = np.array([True, False, True])
x = np.array([10, 20, 30])
result = np.where(cond, x, 0)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('accepts plain number[] for x and y', () => {
    const cond = array([true, false, true]);
    const result = where(cond, [1, 2, 3], [10, 20, 30]) as NDArray;
    const np = runNumPy(`
cond = np.array([True, False, True])
result = np.where(cond, [1, 2, 3], [10, 20, 30])
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('broadcasts a scalar against an array', () => {
    const a = arange(6).reshape(2, 3);
    const result = where(a, a, -1) as NDArray;
    const np = runNumPy(`
a = np.arange(6).reshape(2, 3)
result = np.where(a, a, -1)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('still works with no x/y (returns indices)', () => {
    const a = array([0, 1, 0, 2, 0, 3]);
    const result = where(a) as NDArray[];
    const np = runNumPy(`
a = np.array([0, 1, 0, 2, 0, 3])
result = list(np.where(a))
`);
    expect(result.length).toBe(np.value.length);
    expect(arraysClose(result[0]!.toArray() as number[], np.value[0], 1e-10)).toBe(true);
  });
});
