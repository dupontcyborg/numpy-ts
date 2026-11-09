/**
 * NumPy oracle for view ownership: which free functions and methods return views, what base and
 * the OWNDATA, C_CONTIGUOUS and F_CONTIGUOUS flags report, and whether writes through the result
 * reach the input.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

type Arr = any;

type Input = { name: string; js: () => Arr; py: string };
type Fn = { name: string; js: (a: Arr) => Arr; py: string; writable?: boolean; only?: string[] };

const inputs: Input[] = [
  { name: '1d', js: () => np.arange(6), py: 'np.arange(6.0)' },
  { name: '2d', js: () => np.arange(12).reshape([3, 4]), py: 'np.arange(12.0).reshape(3,4)' },
  { name: '3d', js: () => np.arange(24).reshape([2, 3, 4]), py: 'np.arange(24.0).reshape(2,3,4)' },
  {
    name: '2d transposed',
    js: () => np.arange(12).reshape([3, 4]).T,
    py: 'np.arange(12.0).reshape(3,4).T',
  },
  {
    name: '2d strided window',
    js: () => np.arange(24).reshape([4, 6]).slice('::2', '1:5'),
    py: 'np.arange(24.0).reshape(4,6)[::2, 1:5]',
  },
  {
    name: '2d row range',
    js: () => np.arange(24).reshape([4, 6]).slice('1:3'),
    py: 'np.arange(24.0).reshape(4,6)[1:3]',
  },
  { name: '0d', js: () => np.array(5), py: 'np.array(5.0)' },
  {
    name: 'int32 matrix',
    js: () => np.arange(12, undefined, undefined, 'int32').reshape([4, 3]),
    py: 'np.arange(12, dtype=np.int32).reshape(4,3)',
  },
  {
    name: 'complex matrix',
    js: () =>
      np
        .array(
          Array.from({ length: 12 }, (_, i) => new np.Complex(i, -i)),
          'complex128',
        )
        .reshape([3, 4]),
    py: 'np.array([complex(i,-i) for i in range(12)]).reshape(3,4)',
  },
  { name: 'column', js: () => np.arange(6).reshape([6, 1]), py: 'np.arange(6.0).reshape(6,1)' },
];

const fns: Fn[] = [
  { name: 'np.reshape to 1-D', js: (a) => np.reshape(a, [a.size]), py: 'np.reshape(a, (a.size,))' },
  {
    name: 'np.reshape to 2 rows',
    js: (a) => np.reshape(a, [2, a.size / 2]),
    py: 'np.reshape(a, (2, a.size // 2))',
  },
  { name: 'np.reshape with -1', js: (a) => np.reshape(a, [-1, 1]), py: 'np.reshape(a, (-1, 1))' },
  { name: 'method reshape', js: (a) => a.reshape([a.size]), py: 'a.reshape((a.size,))' },
  {
    name: 'np.reshape of the transpose',
    js: (a) => np.reshape(a.T, [a.size]),
    py: 'np.reshape(a.T, (a.size,))',
  },
  { name: 'np.ravel', js: (a) => np.ravel(a), py: 'np.ravel(a)' },
  { name: 'method ravel', js: (a) => a.ravel(), py: 'a.ravel()' },
  { name: 'np.ravel of the transpose', js: (a) => np.ravel(a.T), py: 'np.ravel(a.T)' },
  { name: 'method flatten', js: (a) => a.flatten(), py: 'a.flatten()' },
  { name: 'np.transpose', js: (a) => np.transpose(a), py: 'np.transpose(a)' },
  { name: 'T', js: (a) => a.T, py: 'a.T' },
  {
    name: 'np.swapaxes',
    js: (a) => np.swapaxes(a, 0, a.ndim - 1),
    py: 'np.swapaxes(a, 0, a.ndim-1)',
  },
  { name: 'np.moveaxis', js: (a) => np.moveaxis(a, 0, -1), py: 'np.moveaxis(a, 0, -1)' },
  { name: 'np.rollaxis', js: (a) => np.rollaxis(a, a.ndim - 1), py: 'np.rollaxis(a, a.ndim-1)' },
  { name: 'np.squeeze', js: (a) => np.squeeze(a), py: 'np.squeeze(a)' },
  {
    name: 'np.squeeze of a padded array',
    js: (a) => np.squeeze(np.expand_dims(a, 0), 0),
    py: 'np.squeeze(np.expand_dims(a, 0), 0)',
  },
  { name: 'np.expand_dims', js: (a) => np.expand_dims(a, 0), py: 'np.expand_dims(a, 0)' },
  { name: 'np.atleast_1d', js: (a) => np.atleast_1d(a), py: 'np.atleast_1d(a)' },
  { name: 'np.atleast_2d', js: (a) => np.atleast_2d(a), py: 'np.atleast_2d(a)' },
  { name: 'np.atleast_3d', js: (a) => np.atleast_3d(a), py: 'np.atleast_3d(a)' },
  {
    name: 'np.atleast_2d with two arguments',
    js: (a) => np.atleast_2d(a, a)[0],
    py: 'np.atleast_2d(a, a)[0]',
  },
  {
    name: 'np.broadcast_to',
    js: (a) => np.broadcast_to(a, [3, ...a.shape]),
    py: 'np.broadcast_to(a, (3,)+a.shape)',
    writable: false,
  },
  {
    name: 'np.broadcast_arrays',
    js: (a) => np.broadcast_arrays(a, np.zeros([3, ...a.shape]))[0],
    py: 'np.broadcast_arrays(a, np.zeros((3,)+a.shape))[0]',
    writable: false,
  },
  {
    name: 'np.broadcast_arrays of an array already at the common shape',
    js: (a) => np.broadcast_arrays(a, np.zeros(a.shape))[0],
    py: 'np.broadcast_arrays(a, np.zeros(a.shape))[0]',
  },
  {
    name: 'np.diagonal',
    js: (a) => np.diagonal(a),
    py: 'np.diagonal(a)',
    only: [
      '2d',
      '3d',
      '2d transposed',
      '2d strided window',
      '2d row range',
      'int32 matrix',
      'complex matrix',
      'column',
    ],
    writable: false,
  },
  {
    name: 'np.diagonal with offset',
    js: (a) => np.diagonal(a, 1),
    py: 'np.diagonal(a, 1)',
    only: ['2d', '3d', '2d transposed', '2d strided window', 'int32 matrix', 'complex matrix'],
    writable: false,
  },
  {
    name: 'method diagonal',
    js: (a) => a.diagonal(),
    py: 'a.diagonal()',
    only: ['2d', '3d', '2d transposed', 'complex matrix'],
    writable: false,
  },
  {
    name: 'np.diag of a matrix',
    js: (a) => np.diag(a),
    py: 'np.diag(a)',
    only: ['2d', '2d transposed', '2d strided window', 'int32 matrix', 'complex matrix'],
    writable: false,
  },
  { name: 'np.diag of a vector', js: (a) => np.diag(a), py: 'np.diag(a)', only: ['1d'] },
  { name: 'np.real', js: (a) => np.real(a), py: 'np.real(a)' },
  {
    name: 'np.imag',
    js: (a) => np.imag(a),
    py: 'np.imag(a)',
    writable: false,
    // the zeros NumPy returns for real input keep the input's memory order, which is not reproduced
    only: [
      '1d',
      '2d',
      '3d',
      '2d strided window',
      '2d row range',
      '0d',
      'int32 matrix',
      'complex matrix',
      'column',
    ],
  },
  {
    name: 'np.split',
    js: (a) => np.split(a, 2)[1],
    py: 'np.split(a, 2)[1]',
    only: ['1d', '2d strided window', '2d row range', 'int32 matrix'],
  },
  {
    name: 'np.array_split',
    js: (a) => np.array_split(a, 3)[1],
    py: 'np.array_split(a, 3)[1]',
    only: ['1d', '2d', '2d transposed', '2d strided window'],
  },
  {
    name: 'np.hsplit',
    js: (a) => np.hsplit(a, 2)[0],
    py: 'np.hsplit(a, 2)[0]',
    only: ['2d', '3d', '2d strided window', 'complex matrix'],
  },
  {
    name: 'np.vsplit',
    js: (a) => np.vsplit(a, 2)[1],
    py: 'np.vsplit(a, 2)[1]',
    only: ['3d', '2d strided window'],
  },
  { name: 'np.dsplit', js: (a) => np.dsplit(a, 2)[1], py: 'np.dsplit(a, 2)[1]', only: ['3d'] },
  { name: 'np.take', js: (a) => np.take(a, [0, 1]), py: 'np.take(a, [0,1])' },
  { name: 'np.tril', js: (a) => np.tril(a), py: 'np.tril(a)', only: ['2d', '3d', '2d transposed'] },
  { name: 'np.triu', js: (a) => np.triu(a), py: 'np.triu(a)', only: ['2d', '3d', '2d transposed'] },
  {
    name: 'np.flip',
    js: (a) => np.flip(a),
    py: 'np.flip(a)',
    only: ['1d', '2d', '3d', '2d transposed', '2d strided window'],
  },
  {
    name: 'np.fliplr',
    js: (a) => np.fliplr(a),
    py: 'np.fliplr(a)',
    only: ['2d', '3d', '2d transposed'],
  },
  {
    name: 'np.flipud',
    js: (a) => np.flipud(a),
    py: 'np.flipud(a)',
    only: ['1d', '2d', '3d', '2d transposed'],
  },
  {
    name: 'np.rot90',
    js: (a) => np.rot90(a),
    py: 'np.rot90(a)',
    only: ['2d', '3d', '2d transposed', 'complex matrix'],
  },
  {
    name: 'np.rot90 with k=0',
    js: (a) => np.rot90(a, 0),
    py: 'np.rot90(a, 0)',
    only: ['2d', '2d transposed'],
  },
  { name: 'np.asarray', js: (a) => np.asarray(a), py: 'np.asarray(a)' },
  {
    name: 'np.ascontiguousarray',
    js: (a) => np.ascontiguousarray(a),
    py: 'np.ascontiguousarray(a)',
  },
  { name: 'np.asfortranarray', js: (a) => np.asfortranarray(a), py: 'np.asfortranarray(a)' },
  { name: 'slice', js: (a) => a.slice('0:2'), py: 'a[0:2]', only: ['1d', '2d', '3d'] },
  { name: 'reversed slice', js: (a) => a.slice('::-1'), py: 'a[::-1]', only: ['1d', '2d', '3d'] },
  {
    name: 'np.unstack',
    js: (a) => np.unstack(a)[0],
    py: 'np.unstack(a)[0]',
    only: ['2d', '3d', '2d transposed'],
  },
  { name: 'view of a view', js: (a) => np.swapaxes(a.T, 0, 1), py: 'np.swapaxes(a.T, 0, 1)' },
];

type Props = {
  same: boolean;
  baseIsNone: boolean;
  ownsData: boolean;
  c: boolean;
  f: boolean;
  shares: boolean;
  baseIsOwner: boolean;
};

const PY_PROPS = `
def props(r, a):
    owner = a.base if a.base is not None else a
    return dict(same=bool(r is a), baseIsNone=bool(r.base is None), ownsData=bool(r.flags.owndata), c=bool(r.flags.c_contiguous), f=bool(r.flags.f_contiguous), shares=bool(np.shares_memory(r, a)), baseIsOwner=bool(r.base is owner))
`;

function jsProps(r: Arr, a: Arr): Props {
  const owner = a.base ?? a;
  return {
    same: r === a,
    baseIsNone: r.base === null,
    ownsData: r.flags.OWNDATA,
    c: r.flags.C_CONTIGUOUS,
    f: r.flags.F_CONTIGUOUS,
    shares: r.size > 0 && a.size > 0 && r.storage.data === a.storage.data,
    baseIsOwner: r.base === owner,
  };
}

function writeFirst(r: Arr, value: number): void {
  r.set(new Array(r.ndim).fill(0), value);
}

describe('NumPy Validation: view ownership', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('base and flags', () => {
    for (const fn of fns) {
      describe(fn.name, () => {
        for (const input of inputs) {
          if (fn.only && !fn.only.includes(input.name)) continue;
          if (!fn.only && input.name === 'column' && fn.name === 'np.diag of a vector') continue;
          it(`on ${input.name}`, () => {
            const py = (() => {
              try {
                return runNumPy(`${PY_PROPS}\na = ${input.py}\nr = ${fn.py}\nresult = props(r, a)`)
                  .value as Props;
              } catch {
                return null;
              }
            })();
            if (py === null) return; // NumPy rejects this combination
            const a = input.js();
            const got = jsProps(fn.js(a), a);
            expect(got).toEqual({
              same: py.same,
              baseIsNone: py.baseIsNone,
              ownsData: py.ownsData,
              c: py.c,
              f: py.f,
              shares: py.shares,
              baseIsOwner: py.baseIsOwner,
            });
          });
        }
      });
    }
  });

  describe('writes through the result', () => {
    for (const fn of fns) {
      if (fn.writable === false) continue;
      for (const input of inputs) {
        if (fn.only && !fn.only.includes(input.name)) continue;
        if (input.name === 'complex matrix' || input.name === '0d') continue;
        it(`${fn.name} on ${input.name} leaves the input as NumPy does`, () => {
          let py: unknown;
          try {
            py = runNumPy(
              `a = ${input.py}\nr = ${fn.py}\nif r.size > 0:\n    r[(0,)*r.ndim] = 99.0\nresult = a.tolist()`,
            ).value;
          } catch {
            return;
          }
          const a = input.js();
          const r = fn.js(a);
          if (r.size > 0) writeFirst(r, 99);
          expect(a.toArray()).toEqual(py);
        });
      }
    }
  });

  describe('reshape of a layout that cannot be a view', () => {
    it('returns a view of a hidden copy of the input shape', () => {
      const a = np.arange(12).reshape([3, 4]);
      const t = a.T;
      const r = np.reshape(t, [12]);
      const py = runNumPy(
        'a = np.arange(12.0).reshape(3,4)\nr = np.reshape(a.T, (12,))\nresult = dict(none=r.base is None, own=bool(r.flags.owndata), baseShape=list(r.base.shape), shares=bool(np.shares_memory(r, a)))',
      ).value as { none: boolean; own: boolean; baseShape: number[]; shares: boolean };
      expect(r.base === null).toBe(py.none);
      expect(r.flags.OWNDATA).toBe(py.own);
      expect([...r.base.shape]).toEqual(py.baseShape);
      expect(r.storage.data === a.storage.data).toBe(py.shares);
    });

    it('reshape keeps views where NumPy can express the new shape with strides', () => {
      for (const [shape, newShape] of [
        [
          [3, 4],
          [4, 3, 1],
        ],
        [
          [4, 6],
          [2, 2, 6],
        ],
        [
          [2, 3, 4],
          [6, 4],
        ],
        [[6], [3, 2]],
      ] as [number[], number[]][]) {
        const size = shape.reduce((x, y) => x * y, 1);
        const a = np.arange(size).reshape(shape).T;
        const r = np.reshape(a, newShape);
        const py = runNumPy(
          `a = np.arange(${size}.0).reshape(${JSON.stringify(shape)}).T\nr = np.reshape(a, ${JSON.stringify(newShape)})\nresult = dict(shares=bool(np.shares_memory(r, a)), values=r.tolist(), strides=[s // 8 for s in r.strides])`,
        ).value as { shares: boolean; values: unknown; strides: number[] };
        expect(r.storage.data === a.storage.data).toBe(py.shares);
        expect(r.toArray()).toEqual(py.values);
        if (py.shares) expect(r.strides.map((s: number) => s)).toEqual(py.strides.map((s) => s));
      }
    });
  });
});
