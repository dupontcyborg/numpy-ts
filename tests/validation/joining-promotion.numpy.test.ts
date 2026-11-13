/**
 * NumPy validation: dtype promotion when joining arrays. concatenate, stack, vstack, hstack,
 * dstack, column_stack, block and append must give NumPy's result_type for any number of inputs,
 * including mixes such as int8, uint16, float32 where promoting pairwise from the left goes wrong.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import {
  append,
  block,
  column_stack,
  concatenate,
  dstack,
  hstack,
  promote_types,
  stack,
  vstack,
} from '../../src';
import type { DType } from '../../src/common/dtype';
import { DTYPES, expectSame, lcg, sample, toJs, toPy } from './_dtype-grid';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

const JOINERS: [string, (xs: any[]) => any, string][] = [
  ['concatenate', (xs) => concatenate(xs), 'np.concatenate(xs)'],
  ['concatenate axis=1', (xs) => concatenate(xs, 1), 'np.concatenate(xs, axis=1)'],
  ['concatenate axis=null', (xs) => concatenate(xs, null), 'np.concatenate(xs, axis=None)'],
  ['stack', (xs) => stack(xs), 'np.stack(xs)'],
  ['stack axis=2', (xs) => stack(xs, 2), 'np.stack(xs, axis=2)'],
  ['vstack', (xs) => vstack(xs), 'np.vstack(xs)'],
  ['hstack', (xs) => hstack(xs), 'np.hstack(xs)'],
  ['dstack', (xs) => dstack(xs), 'np.dstack(xs)'],
  ['column_stack', (xs) => column_stack(xs), 'np.column_stack(xs)'],
  ['block (flat)', (xs) => block(xs), 'np.block(xs)'],
  ['block (rows)', (xs) => block(xs.map((x) => [x])), 'np.block([[x] for x in xs])'],
  ['block (grid)', (xs) => block([xs, xs]), 'np.block([xs, xs])'],
];

function join(dtypes: DType[], rng: () => number, label: string) {
  const samples = dtypes.map((d) => sample(rng, d, [2, 3]));
  const xs = samples.map((s, i) => toJs(s, dtypes[i]!));
  const py = `xs = [${samples.map((s, i) => toPy(s, dtypes[i]!)).join(', ')}]\n`;
  for (const [name, fn, expr] of JOINERS) {
    expectSame(fn(xs), runNumPy(`${py}result = ${expr}`), `${name} ${label} ${dtypes.join(',')}`);
  }
}

describe('NumPy Validation: dtype promotion when joining arrays', { timeout: 120_000 }, () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('promotes a mix that a left fold gets wrong (int8, uint16, float32)', () => {
    const rng = lcg(1);
    join(['int8', 'uint16', 'float32'], rng, 'fold');
    join(['uint16', 'int8', 'float32'], rng, 'fold');
    join(['float32', 'int8', 'uint16'], rng, 'fold');
    join(['int8', 'uint64', 'float16', 'int32'], rng, 'fold');
    expect(
      concatenate([
        toJs(sample(rng, 'int8', [2]), 'int8'),
        toJs(sample(rng, 'uint16', [2]), 'uint16'),
        toJs(sample(rng, 'float32', [2]), 'float32'),
      ]).dtype,
    ).toBe('float32');
  });

  it('matches NumPy for random triples and quadruples of dtypes', () => {
    const rng = lcg(2);
    for (let i = 0; i < 90; i++) {
      const n = 3 + Math.floor(rng() * 2);
      const dtypes = Array.from({ length: n }, () => DTYPES[Math.floor(rng() * DTYPES.length)]!);
      join(dtypes, rng, `case ${i}`);
    }
  });

  it('matches NumPy for every dtype pair', () => {
    const rng = lcg(3);
    for (const da of DTYPES) {
      for (const db of DTYPES) {
        const sa = sample(rng, da, [2, 3]);
        const sb = sample(rng, db, [2, 3]);
        const a = toJs(sa, da);
        const b = toJs(sb, db);
        const py = `a = ${toPy(sa, da)}\nb = ${toPy(sb, db)}\n`;
        expectSame(
          concatenate([a, b]),
          runNumPy(`${py}result = np.concatenate([a, b])`),
          `concatenate ${da}/${db}`,
        );
        expectSame(append(a, b), runNumPy(`${py}result = np.append(a, b)`), `append ${da}/${db}`);
        expectSame(
          append(a, b, 0),
          runNumPy(`${py}result = np.append(a, b, axis=0)`),
          `append axis ${da}/${db}`,
        );
        expect(concatenate([a, b]).dtype).toBe(promote_types(da, db));
      }
    }
  });

  it('promotes at each level of a nested block, as NumPy does for small inputs', () => {
    const rng = lcg(4);
    for (let i = 0; i < 60; i++) {
      const dtypes = Array.from({ length: 4 }, () => DTYPES[Math.floor(rng() * DTYPES.length)]!);
      const samples = dtypes.map((d) => sample(rng, d, [2, 2]));
      const xs = samples.map((s, k) => toJs(s, dtypes[k]!));
      const py = `xs = [${samples.map((s, k) => toPy(s, dtypes[k]!)).join(', ')}]\n`;
      expectSame(
        block([
          [xs[0], xs[1]],
          [xs[2], xs[3]],
        ]),
        runNumPy(`${py}result = np.block([[xs[0], xs[1]], [xs[2], xs[3]]])`),
        `block 2x2 ${dtypes.join(',')}`,
      );
    }
    const a = toJs({ re: [1, 2], shape: [2] }, 'int8');
    const b = toJs({ re: [1, 2], shape: [2] }, 'uint16');
    const c = toJs({ re: [1.5, 2, 3, 4], shape: [4] }, 'float32');
    expectSame(
      block([[a, b], [c]]),
      runNumPy(`a = np.array([1, 2], 'int8')
b = np.array([1, 2], 'uint16')
c = np.array([1.5, 2, 3, 4], 'float32')
result = np.block([[a, b], [c]])`),
      'ragged block',
    );
  });

  it('keeps values exact when widening 64-bit integers', () => {
    const a = toJs({ re: [1, 2], shape: [2] }, 'int32');
    expect(concatenate([a, toJs({ re: [3], shape: [1] }, 'int64')]).dtype).toBe('int64');
    expect(concatenate([a, toJs({ re: [3], shape: [1] }, 'uint64')]).dtype).toBe('float64');
    expect(
      concatenate([toJs({ re: [3], shape: [1] }, 'uint8'), toJs({ re: [3], shape: [1] }, 'uint64')])
        .dtype,
    ).toBe('uint64');
    expect(
      concatenate([
        toJs({ re: [3], shape: [1] }, 'uint64'),
        toJs({ re: [-1], shape: [1] }, 'int64'),
      ]).dtype,
    ).toBe('float64');
  });
});
