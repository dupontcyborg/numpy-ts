/**
 * NumPy validation tests for np.interp — focuses on the `period` kwarg.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { array, interp } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: interp', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('regression: works without period', () => {
    const x = array([1.5, 2.5, 3.5]);
    const xp = array([1.0, 2.0, 3.0, 4.0]);
    const fp = array([10.0, 20.0, 30.0, 40.0]);
    const result = interp(x, xp, fp);
    const np = runNumPy(`
x = np.array([1.5, 2.5, 3.5])
xp = np.array([1.0, 2.0, 3.0, 4.0])
fp = np.array([10.0, 20.0, 30.0, 40.0])
result = np.interp(x, xp, fp)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('period=2*pi wraps at the boundary', () => {
    // Interpolating sin samples on a periodic interval.
    const xp = array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const fp = array([0.84, 0.91, 0.14, -0.76, -0.96]);
    const x = array([-180.0, -90.0, 0.0, 90.0, 180.0]).multiply(Math.PI / 180);
    const period = 2 * Math.PI;
    const result = interp(x, xp, fp, undefined, undefined, period);
    const np = runNumPy(`
xp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fp = np.array([0.84, 0.91, 0.14, -0.76, -0.96])
x = np.array([-180.0, -90.0, 0.0, 90.0, 180.0]) * np.pi / 180
result = np.interp(x, xp, fp, period=2*np.pi)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('period handles unsorted xp by sorting internally', () => {
    const xp = array([3.0, 1.0, 2.0, 4.0]);
    const fp = array([30.0, 10.0, 20.0, 40.0]);
    const x = array([0.5, 1.5, 2.5, 3.5, 4.5]);
    const result = interp(x, xp, fp, undefined, undefined, 5);
    const np = runNumPy(`
xp = np.array([3.0, 1.0, 2.0, 4.0])
fp = np.array([30.0, 10.0, 20.0, 40.0])
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
result = np.interp(x, xp, fp, period=5)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('left/right are ignored when period is provided', () => {
    const xp = array([1.0, 2.0, 3.0]);
    const fp = array([10.0, 20.0, 30.0]);
    const x = array([-100.0, 100.0]); // far outside [0, period); should still wrap
    const result = interp(x, xp, fp, 999, 999, 4);
    const np = runNumPy(`
xp = np.array([1.0, 2.0, 3.0])
fp = np.array([10.0, 20.0, 30.0])
x = np.array([-100.0, 100.0])
result = np.interp(x, xp, fp, left=999, right=999, period=4)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('period=0 throws', () => {
    const x = array([1.0]);
    const xp = array([0.0, 1.0]);
    const fp = array([0.0, 1.0]);
    expect(() => interp(x, xp, fp, undefined, undefined, 0)).toThrow();
  });
});
