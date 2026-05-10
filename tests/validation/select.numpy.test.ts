/**
 * NumPy validation tests for np.select — verifies array-like condlist/choicelist
 * and array-valued default support.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { arange, array, select } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: select', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('regression: scalar default with NDArray inputs', () => {
    const x = arange(10);
    const condlist = [x.less(3), x.greater(5)];
    const choicelist = [x.multiply(10), x.multiply(-1)];
    const result = select(condlist, choicelist, -99);
    const np = runNumPy(`
x = np.arange(10)
result = np.select([x < 3, x > 5], [x*10, -x], default=-99)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('accepts plain number[] in condlist/choicelist', () => {
    const result = select(
      [
        [true, false, true, false],
        [false, true, false, true],
      ],
      [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
      ],
      0,
    );
    const np = runNumPy(`
result = np.select([[True, False, True, False], [False, True, False, True]],
                   [[1, 2, 3, 4], [10, 20, 30, 40]], default=0)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('accepts an array-valued default (broadcasts against choices)', () => {
    const x = arange(6);
    const result = select([x.less(2), x.greater(4)], [x.multiply(100), x.multiply(-1)], x);
    const np = runNumPy(`
x = np.arange(6)
result = np.select([x < 2, x > 4], [x*100, -x], default=x)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('plain number[] for default broadcasts elementwise', () => {
    const cond = array([true, false, true, false]);
    const result = select([cond], [[10, 20, 30, 40]], [1, 2, 3, 4]);
    const np = runNumPy(`
result = np.select([np.array([True, False, True, False])],
                   [[10, 20, 30, 40]], default=[1, 2, 3, 4])
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });
});
