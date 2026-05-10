/**
 * NumPy validation tests for np.block — verifies the recursive nested-list
 * semantics match NumPy.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { block, empty } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: block', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  it('matches numpy on the canonical 2x2 nested example', () => {
    const a = empty([2, 3]);
    a.fill(1);
    const b = empty([2, 2]);
    b.fill(2);
    const c = empty([1, 1]);
    c.fill(3);
    const d = empty([1, 4]);
    d.fill(4);

    const result = block([
      [a, b],
      [c, d],
    ]);

    const np = runNumPy(`
a = np.empty([2, 3]); a.fill(1)
b = np.empty([2, 2]); b.fill(2)
c = np.empty([1, 1]); c.fill(3)
d = np.empty([1, 4]); d.fill(4)
result = np.block([[a, b], [c, d]])
`);

    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('matches numpy on a flat list (concat along axis -1)', () => {
    const a = empty([2, 3]);
    a.fill(1);
    const b = empty([2, 2]);
    b.fill(2);

    const result = block([a, b]);

    const np = runNumPy(`
a = np.empty([2, 3]); a.fill(1)
b = np.empty([2, 2]); b.fill(2)
result = np.block([a, b])
`);

    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('matches numpy on a 3-deep nested example', () => {
    const a = empty([1, 2, 2]);
    a.fill(1);
    const b = empty([1, 2, 2]);
    b.fill(2);
    const c = empty([1, 2, 4]);
    c.fill(3);
    const d = empty([1, 2, 4]);
    d.fill(4);

    const result = block([
      [[a, b], [c]],
      [[c], [a, b]],
    ]);

    const np = runNumPy(`
a = np.empty([1, 2, 2]); a.fill(1)
b = np.empty([1, 2, 2]); b.fill(2)
c = np.empty([1, 2, 4]); c.fill(3)
d = np.empty([1, 2, 4]); d.fill(4)
result = np.block([[[a, b], [c]], [[c], [a, b]]])
`);

    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });
});
