/**
 * NumPy validation: all-boolean JavaScript data infers dtype bool in array and asarray.
 * Mixed data keeps the library defaults and is covered by unit tests, not here.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import {
  array,
  asarray,
  bitwise_and,
  compress,
  invert,
  logical_not,
  select,
  sum,
  where,
} from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

const flat = (a: { toArray(): unknown }): unknown => JSON.stringify(a.toArray());
const asNumbers = (v: unknown): unknown =>
  Array.isArray(v) ? v.map(asNumbers) : typeof v === 'boolean' ? Number(v) : v;

describe('NumPy Validation: bool dtype inference', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  const cases: [string, unknown][] = [
    ['1d', [true, false]],
    ['single true', [true]],
    ['single false', [false]],
    ['all true', [true, true, true, true]],
    [
      '2d',
      [
        [true, false],
        [false, true],
      ],
    ],
    ['column', [[true], [false], [true]]],
    ['3d', [[[true, false]], [[false, false]]]],
    ['4d', [[[[true]]]]],
    ['0-d true', true],
    ['0-d false', false],
  ];

  for (const [name, data] of cases) {
    it(`array(${name}) matches NumPy dtype, shape and values`, () => {
      const np_ = runNumPy(
        `result = np.array(${JSON.stringify(data).replace(/true/g, 'True').replace(/false/g, 'False')})`,
      );
      for (const result of [array(data), asarray(data)]) {
        expect(result.dtype).toBe(np_.dtype);
        expect(Array.from(result.shape)).toEqual(np_.shape);
        expect(asNumbers(result.toArray())).toEqual(asNumbers(np_.value));
      }
    });
  }

  it('empty data still infers float64', () => {
    for (const [data, py] of [
      [[], '[]'],
      [[[], []], '[[], []]'],
    ] as const) {
      const r = array(data);
      const np_ = runNumPy(`result = np.array(${py})`);
      expect(r.dtype).toBe(np_.dtype);
      expect(Array.from(r.shape)).toEqual(np_.shape);
    }
  });

  it('explicit dtype is unaffected', () => {
    const r = array([true, false, true], 'float64');
    const np_ = runNumPy('result = np.array([True, False, True], dtype="float64")');
    expect(r.dtype).toBe(np_.dtype);
    expect(asNumbers(r.toArray())).toEqual(np_.value);
    const i = array([true, false], 'int32');
    expect(i.dtype).toBe('int32');
    expect(Array.from(i.toArray() as number[])).toEqual([1, 0]);
  });

  it('boolean masks built from literals drive NumPy-equivalent results', () => {
    const m = array([true, false, true]);
    const v = array([1, 2, 3], 'int32');
    const w = array([true, true, false]);
    const np_ = runNumPy(`
m = np.array([True, False, True])
v = np.array([1, 2, 3], dtype='int32')
w = np.array([True, True, False])
result = {
  'compress': np.compress(m, v).tolist(),
  'invert': np.invert(m).astype(int).tolist(),
  'not': np.logical_not(m).astype(int).tolist(),
  'and': np.bitwise_and(m, w).astype(int).tolist(),
  'sum': int(np.sum(m)),
  'where': np.where(m, v, np.array([9, 9, 9], dtype='int32')).tolist(),
  'select': np.select([m], [v]).tolist(),
}
`);
    const expected = np_.value as Record<string, unknown>;
    expect(JSON.parse(flat(compress(m, v)) as string)).toEqual(expected.compress);
    expect(asNumbers(invert(m).toArray())).toEqual(expected.invert);
    expect(asNumbers(logical_not(m).toArray())).toEqual(expected.not);
    expect(asNumbers(bitwise_and(m, w).toArray())).toEqual(expected.and);
    expect(Number(sum(m))).toBe(expected.sum);
    expect(Array.from(where(m, v, array([9, 9, 9], 'int32')).toArray() as number[])).toEqual(
      expected.where as number[],
    );
    expect(Array.from(select([m], [v]).toArray() as number[])).toEqual(expected.select as number[]);
  });
});
