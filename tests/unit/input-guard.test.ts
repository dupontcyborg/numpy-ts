/**
 * Unit tests for the non-NDArray input guard
 *
 * Functions that take an NDArray must reject plain arrays and scalars with a TypeError instead of
 * reading undefined shape and size and allocating NaN-sized storage.
 */

import { describe, expect, it } from 'vitest';
import {
  add,
  append,
  apply_along_axis,
  array,
  array_equal,
  bincount,
  byteswap,
  compress,
  convolve,
  copy,
  correlate,
  delete_,
  diag,
  diagflat,
  diff,
  digitize,
  einsum_path,
  empty,
  empty_like,
  fill,
  full_like,
  histogram,
  isin,
  item,
  ix_,
  fft as npFft,
  ones,
  ones_like,
  pad,
  place,
  polyadd,
  polyfit,
  polyval,
  put,
  putmask,
  random,
  ravel_multi_index,
  require,
  roots,
  searchsorted,
  select,
  sum,
  take,
  tobytes,
  tolist,
  tril,
  triu,
  view,
  where,
  zeros,
  zeros_like,
} from '../../src';
import { ArrayStorage } from '../../src/common/storage';
import * as core from '../../src/core';
import { toStorage } from '../../src/core/types';

// Deliberately bypasses the NDArray parameter types.
const bad = (v: unknown): any => v;

describe('non-NDArray input guard', () => {
  it.each([
    ['add', () => add(bad([1, 2]), bad([3, 4]))],
    ['sum', () => sum(bad([1, 2]))],
    ['correlate', () => correlate(bad([1, 2, 3]), bad([0, 1]))],
    ['convolve', () => convolve(bad([1, 2, 3]), bad([0, 1]))],
    ['searchsorted', () => searchsorted(bad([1, 2, 3]), bad(2))],
    ['ix_', () => ix_(bad([0, 1]), bad([1, 2]))],
    [
      'ravel_multi_index',
      () =>
        ravel_multi_index(
          bad([
            [0, 1],
            [1, 0],
          ]),
          [2, 2],
        ),
    ],
    ['array_equal', () => array_equal(bad([1, 2]), bad([1, 3]))],
    ['isin', () => isin(array([1, 2, 3]), bad([2, 3]))],
    ['compress', () => compress(bad([true, false]), array([1, 2]))],
  ])('%s throws a TypeError for a plain array', (_name, call) => {
    expect(call).toThrow(TypeError);
    expect(call).toThrow(/np\.array/);
  });

  it('names what was passed', () => {
    expect(() => add(bad([1]), array([1]))).toThrow(/plain array/);
    expect(() => add(bad(null), array([1]))).toThrow(/null/);
    expect(() => add(bad('x'), array([1]))).toThrow(/string/);
    expect(() => add(bad({}), array([1]))).toThrow(/Object/);
    expect(() => add(bad(Object.create(null)), array([1]))).toThrow(/an object/);
  });

  it('passes a raw ArrayStorage through unchanged', () => {
    const storage = ArrayStorage.zeros([2], 'float64');
    expect(toStorage(storage)).toBe(storage);
  });

  it('still accepts an NDArray', () => {
    expect(add(array([1, 2]), array([3, 4])).toArray()).toEqual([4, 6]);
  });

  it('keeps the scalar and shape forms that functions take on purpose', () => {
    const a = array([1, 2, 3]);
    put(a, [0, 2], 9);
    expect(a.toArray()).toEqual([9, 2, 9]);
    put(a, [0, 1], [7, 8]);
    expect(a.toArray()).toEqual([7, 8, 9]);
    expect(
      apply_along_axis(
        () => 5,
        0,
        array([
          [1, 2],
          [3, 4],
        ]),
      ).toArray(),
    ).toEqual([5, 5]);
    const [path] = einsum_path('ij,jk->ik', [10, 5], [5, 8]);
    expect(Array.isArray(path)).toBe(true);
  });
});

describe('guard on functions that call NDArray members directly', () => {
  const a2 = () =>
    array([
      [1, 2],
      [3, 4],
    ]);

  it.each([
    ['copy', () => copy(bad([1, 2]))],
    ['view', () => view(bad([1, 2]))],
    ['byteswap', () => byteswap(bad([1, 2]))],
    ['tobytes', () => tobytes(bad([1, 2]))],
    ['tolist', () => tolist(bad([1, 2]))],
    ['item', () => item(bad([1]))],
    ['fill', () => fill(bad([1, 2]), 0)],
    ['delete_', () => delete_(bad([1, 2, 3]), 0)],
    ['diag', () => diag(bad([1, 2]))],
    ['diagflat', () => diagflat(bad([1, 2]))],
    ['tril', () => tril(bad([[1, 2]]))],
    ['triu', () => triu(bad([[1, 2]]))],
    ['pad', () => pad(bad([1, 2]), 1)],
    ['polyfit x', () => polyfit(bad([1, 2, 3]), array([1, 2, 3]), 1)],
    ['polyfit y', () => polyfit(array([1, 2, 3]), bad([1, 2, 3]), 1)],
    ['require', () => require(bad([1, 2]))],
    ['zeros_like', () => zeros_like(bad([1, 2]))],
    ['ones_like', () => ones_like(bad([1, 2]))],
    ['empty_like', () => empty_like(bad([1, 2]))],
    ['full_like', () => full_like(bad([1, 2]), 7)],
  ])('%s throws a TypeError for a plain array', (_name, call) => {
    expect(call).toThrow(TypeError);
    expect(call).toThrow(/np\.array/);
  });

  it('still works on an NDArray', () => {
    expect(copy(array([1, 2])).toArray()).toEqual([1, 2]);
    expect(view(array([1, 2])).toArray()).toEqual([1, 2]);
    expect(byteswap(array([1], 'uint8')).toArray()).toEqual([1]);
    expect(tobytes(array([1], 'uint8'))).toEqual(new Uint8Array([1]));
    expect(tolist(a2())).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(item(array([5]))).toBe(5);
    const f = array([1, 2]);
    fill(f, 9);
    expect(f.toArray()).toEqual([9, 9]);
    expect(delete_(array([1, 2, 3]), 1).toArray()).toEqual([1, 3]);
    expect(diag(array([1, 2])).toArray()).toEqual([
      [1, 0],
      [0, 2],
    ]);
    expect(diagflat(a2()).shape).toEqual([4, 4]);
    expect(tril(a2()).toArray()).toEqual([
      [1, 0],
      [3, 4],
    ]);
    expect(triu(a2()).toArray()).toEqual([
      [1, 2],
      [0, 4],
    ]);
    expect(pad(array([1, 2]), 1).toArray()).toEqual([0, 1, 2, 0]);
    expect(polyfit(array([0, 1, 2]), array([1, 3, 5]), 1).toArray()[0]).toBeCloseTo(2);
    expect(require(array([1, 2]), 'float32').dtype).toBe('float32');
    expect(zeros_like(a2()).toArray()).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(ones_like(array([1, 2])).toArray()).toEqual([1, 1]);
    expect(empty_like(a2()).shape).toEqual([2, 2]);
    expect(full_like(array([1, 2]), 7).toArray()).toEqual([7, 7]);
  });
});

describe('random.choice, permutation and shuffle population guard', () => {
  it('rejects a plain array population', () => {
    expect(() => random.choice(bad([1, 2, 3]), 2)).toThrow(TypeError);
    expect(() => random.choice(bad([1, 2, 3]), 2)).toThrow(/np\.array/);
    expect(() => random.default_rng(1).choice(bad([1, 2, 3]), 2)).toThrow(/np\.array/);
    expect(() => random.permutation(bad([1, 2, 3]))).toThrow(/np\.array/);
    expect(() => random.default_rng(1).permutation(bad([1, 2, 3]))).toThrow(/np\.array/);
    expect(() => random.shuffle(bad([1, 2, 3]))).toThrow(/np\.array/);
    expect(() => random.default_rng(1).shuffle(bad([1, 2, 3]))).toThrow(/np\.array/);
  });

  it('still draws from an NDArray or an integer population', () => {
    const pop = array([10, 20, 30]);
    const members = new Set([10, 20, 30]);
    const withReplace = random.choice(pop, 5);
    expect(withReplace.shape).toEqual([5]);
    for (const v of withReplace.toArray() as number[]) expect(members.has(v)).toBe(true);
    const noReplace = random.choice(pop, 3, false).toArray() as number[];
    expect([...noReplace].sort((x, y) => x - y)).toEqual([10, 20, 30]);
    expect(members.has(random.choice(pop) as number)).toBe(true);
    expect(random.choice(4, 3).shape).toEqual([3]);
    expect(random.choice(4)).toBeLessThan(4);
    const weighted = random.choice(pop, 4, true, [0, 0, 1]).toArray() as number[];
    expect(weighted).toEqual([30, 30, 30, 30]);
    expect(random.default_rng(1).choice(pop, 2, false).shape).toEqual([2]);
    expect(random.permutation(pop).shape).toEqual([3]);
    expect(random.permutation(3).shape).toEqual([3]);
    const s = array([1, 2, 3]);
    random.shuffle(s);
    expect([...(s.toArray() as number[])].sort()).toEqual([1, 2, 3]);
  });
});

describe('plain-array forms that must keep working', () => {
  it('bincount, digitize and histogram take array_like data', () => {
    expect(
      Array.from(bincount([0, 1, 1, 3]).toArray() as ArrayLike<number | bigint>, Number),
    ).toEqual([1, 2, 0, 1]);
    expect(
      Array.from(
        digitize([0.5, 1.5, 2.5], [0, 1, 2, 3]).toArray() as ArrayLike<number | bigint>,
        Number,
      ),
    ).toEqual([1, 2, 3]);
    const [counts, edges] = histogram([1, 2, 1], 2);
    expect(Array.from(counts.toArray() as ArrayLike<number | bigint>, Number)).toEqual([2, 1]);
    expect(edges.toArray()).toEqual([1, 1.5, 2]);
  });

  it('take indices, select, and where x/y accept plain values', () => {
    expect(take(array([10, 20, 30]), [2, 0]).toArray()).toEqual([30, 10]);
    expect(
      select([array([true, false, false]), array([false, true, false])], [1, 2], 0).toArray(),
    ).toEqual([1, 2, 0]);
    const cond = array([true, false, true]);
    expect(where(cond, [1, 2, 3], [4, 5, 6]).toArray()).toEqual([1, 5, 3]);
    expect(where(cond, 1, 0).toArray()).toEqual([1, 0, 1]);
  });

  it('place, putmask, put and append take plain values', () => {
    const a = array([1, 2, 3, 4]);
    place(a, array([true, false, true, false]), [8, 9]);
    expect(a.toArray()).toEqual([8, 2, 9, 4]);
    const b = array([1, 2, 3, 4]);
    putmask(b, array([false, true, false, true]), [7, 6]);
    expect(b.toArray()).toEqual([1, 6, 3, 6]);
    const c = array([1, 2, 3]);
    put(c, [0, 2], [5, 6]);
    expect(c.toArray()).toEqual([5, 2, 6]);
    expect(append(array([1, 2]), [3, 4]).toArray()).toEqual([1, 2, 3, 4]);
  });

  it('diff prepend and sum where take plain values', () => {
    expect(diff(array([1, 3, 6]), 1, -1, [0]).toArray()).toEqual([1, 2, 3]);
    expect(sum(array([1, 2, 3]), undefined, false, { where: [true, false, true] })).toBe(4);
  });

  it('polynomial helpers take plain coefficients', () => {
    expect(polyadd([1, 2], [1, 1, 1]).toArray()).toEqual([1, 2, 3]);
    expect(polyval([1, 2, 3], 2)).toBe(11);
    const r = (roots([1, -3, 2]).toArray() as { re: number; im: number }[]).map((z) => z.re);
    expect(r.sort((x, y) => x - y)).toEqual([1, 2]);
  });

  it('binary ops take a scalar operand', () => {
    expect(add(array([1, 2]), 2).toArray()).toEqual([3, 4]);
  });
});

describe('invalid shape guard', () => {
  it.each([
    ['zeros', zeros],
    ['ones', ones],
    ['empty', empty],
  ])('%s rejects dimensions that are not non-negative integers', (_name, create) => {
    expect(() => create(bad([2, Number.NaN]))).toThrow(TypeError);
    expect(() => create(bad([1.5]))).toThrow(/non-negative integers/);
    expect(() => create([-1])).toThrow(TypeError);
    expect(() => create(bad([[5, 5]]))).toThrow(TypeError);
  });

  it('rejects a shape passed as an array to rand, which takes separate sizes', () => {
    expect(() => random.rand(bad([5, 5]))).toThrow(TypeError);
  });

  it('still allocates valid shapes, including empty and scalar ones', () => {
    expect(zeros([0, 3]).shape).toEqual([0, 3]);
    expect(ones([]).shape).toEqual([]);
    expect(empty([2, 2]).shape).toEqual([2, 2]);
    expect(random.rand(2, 3)).toBeDefined();
  });
});

describe('guard through the core entry point', () => {
  const c = core as any;

  it('rejects plain arrays in array functions', () => {
    expect(() => c.add([1, 2], [3, 4])).toThrow(/np\.array/);
    expect(() => c.copy([1, 2])).toThrow(TypeError);
  });

  it('rejects plain arrays as random populations', () => {
    expect(() => c.random.choice([1, 2, 3], 2)).toThrow(/np\.array/);
    expect(() => c.random.permutation([1, 2, 3])).toThrow(TypeError);
    expect(() => c.random.shuffle([1, 2, 3])).toThrow(TypeError);
    expect(() => c.random.default_rng(1).choice([1, 2, 3], 2)).toThrow(TypeError);
  });

  it('still samples from an integer or an ArrayStorage population', () => {
    const pop = ArrayStorage.zeros([4], 'float64');
    expect(c.random.choice(pop, 2).size).toBe(2);
    expect(c.random.choice(5, 3).size).toBe(3);
    expect(c.random.permutation(pop).size).toBe(4);
    expect(c.random.permutation(4).size).toBe(4);
    expect(() => c.random.shuffle(pop)).not.toThrow();
  });

  it('samples from an NDArrayCore population, with and without probabilities', () => {
    const nd = c.array([1, 2, 3, 4]);
    expect(c.random.choice(nd, 2).size).toBe(2);
    expect(c.random.choice(nd, 2, true, c.array([0, 0, 0, 1])).toArray()).toEqual([4, 4]);
    expect(c.random.permutation(nd).size).toBe(4);
    expect(c.random.default_rng(1).choice(nd, 2).size).toBe(2);
    expect(c.random.default_rng(1).permutation(nd).size).toBe(4);
    expect(() => c.random.shuffle(nd)).not.toThrow();
    expect(() => c.random.default_rng(1).shuffle(nd)).not.toThrow();
  });
});

describe('fft guard', () => {
  const arrayFns = [
    'fft',
    'ifft',
    'fft2',
    'ifft2',
    'fftn',
    'ifftn',
    'rfft',
    'irfft',
    'rfft2',
    'irfft2',
    'rfftn',
    'irfftn',
    'hfft',
    'ihfft',
    'fftshift',
    'ifftshift',
  ] as const;
  const entries = [
    ['numpy-ts', npFft as any],
    ['numpy-ts/core', core.fft as any],
  ] as const;

  for (const [label, ns] of entries) {
    describe(label, () => {
      it.each(arrayFns)('%s rejects a plain array', (name) => {
        expect(() => ns[name]([1, 2, 3, 4])).toThrow(TypeError);
        expect(() => ns[name]([1, 2, 3, 4])).toThrow(/np\.array/);
        expect(() =>
          ns[name]([
            [1, 2],
            [3, 4],
          ]),
        ).toThrow(/np\.array/);
        expect(() => ns[name](3)).toThrow(/np\.array/);
      });

      it('still transforms an NDArray', () => {
        const x = array([1, 2, 3, 4]);
        expect(ns.fft(x).shape).toEqual([4]);
        expect(ns.ifft(ns.fft(x)).shape).toEqual([4]);
        expect(ns.rfft(x).shape).toEqual([3]);
        expect(
          ns.fft2(
            array([
              [1, 2],
              [3, 4],
            ]),
          ).shape,
        ).toEqual([2, 2]);
        expect(ns.fftshift(x).toArray()).toEqual([3, 4, 1, 2]);
        expect(ns.ifftshift(array([1, 2, 3, 4, 5])).toArray()).toEqual([3, 4, 5, 1, 2]);
      });

      it('takes plain numbers in the frequency helpers', () => {
        expect(ns.fftfreq(4).shape).toEqual([4]);
        expect(ns.rfftfreq(4).shape).toEqual([3]);
      });
    });
  }

  it('still accepts a raw ArrayStorage at runtime', () => {
    const st = ArrayStorage.zeros([4], 'float64');
    expect((core.fft as any).fft(st).size).toBe(4);
    expect((npFft as any).fft(st).size).toBe(4);
  });
});

describe('random inputs given as arrays', () => {
  const c = core as any;

  it('core multinomial, dirichlet and multivariate_normal unwrap an NDArrayCore', () => {
    expect(c.random.multinomial(10, c.array([0.5, 0.5]), 3).shape).toEqual([3, 2]);
    expect(c.random.multinomial(10, c.array([0, 1])).toArray()).toEqual([0n, 10n]);
    expect(c.random.dirichlet(c.array([1, 2, 3]), 4).shape).toEqual([4, 3]);
    const mvn = c.random.multivariate_normal(
      c.array([0, 0]),
      c.array([
        [1, 0],
        [0, 1],
      ]),
      5,
    );
    expect(mvn.shape).toEqual([5, 2]);
  });

  it('core accepts plain number arrays and a raw ArrayStorage for the same parameters', () => {
    expect(c.random.multinomial(10, [0, 1]).toArray()).toEqual([0n, 10n]);
    expect(c.random.dirichlet([1, 2], 2).shape).toEqual([2, 2]);
    expect(
      c.random.multivariate_normal(
        [0, 0],
        [
          [1, 0],
          [0, 1],
        ],
        3,
      ).shape,
    ).toEqual([3, 2]);
    const pvals = ArrayStorage.fromData(new Float64Array([0, 1]), [2], 'float64');
    expect(c.random.multinomial(10, pvals).toArray()).toEqual([0n, 10n]);
    const alpha = ArrayStorage.fromData(new Float64Array([1, 2]), [2], 'float64');
    expect(c.random.dirichlet(alpha, 2).shape).toEqual([2, 2]);
  });

  it('core returns an NDArrayCore for a sized sample and a number otherwise', () => {
    expect(c.random.random([2]).shape).toEqual([2]);
    expect(typeof c.random.random()).toBe('number');
    expect(c.random.rand(2, 3).shape).toEqual([2, 3]);
    expect(typeof c.random.randn()).toBe('number');
    expect(c.random.default_rng(1).integers(0, 5, [3]).shape).toEqual([3]);
  });

  it('main multinomial, dirichlet and multivariate_normal take an NDArray', () => {
    expect(random.multinomial(10, array([0, 1])).toArray()).toEqual([0n, 10n]);
    expect(random.dirichlet(array([1, 2]), 2).shape).toEqual([2, 2]);
    expect(
      random.multivariate_normal(
        array([0, 0]),
        array([
          [1, 0],
          [0, 1],
        ]),
        3,
      ).shape,
    ).toEqual([3, 2]);
  });
});
