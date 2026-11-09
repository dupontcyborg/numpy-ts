/**
 * Printing of non-contiguous views, ownership flags of view-returning functions, and the
 * utility float formatters. Expected strings are what NumPy prints.
 */

import { afterEach, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { array2string as array2stringOps } from '../../src/common/ops/formatting';
import { ArrayStorage } from '../../src/common/storage';

afterEach(() => {
  np.set_printoptions({
    precision: 8,
    threshold: 1000,
    edgeitems: 3,
    linewidth: 75,
    suppress: false,
    floatmode: 'maxprec',
    sign: '-',
  });
});

describe('printing non-contiguous views', () => {
  it('prints a transposed matrix in logical order', () => {
    const a = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.T.toString()).toBe('[[1. 4.]\n [2. 5.]\n [3. 6.]]');
    expect(np.array_repr(a.T)).toBe('array([[1., 4.],\n       [2., 5.],\n       [3., 6.]])');
  });

  it('prints reversed and strided slices without reading outside the view', () => {
    const a = np.arange(12).reshape([3, 4]);
    expect(a.slice('::-1', ':').toString()).toBe(
      '[[ 8.  9. 10. 11.]\n [ 4.  5.  6.  7.]\n [ 0.  1.  2.  3.]]',
    );
    expect(np.arange(10).slice('1::2').toString()).toBe('[1. 3. 5. 7. 9.]');
  });

  it('prints the real and imaginary parts of a complex array', () => {
    const z = np.array([new np.Complex(1, 2), new np.Complex(3, 4)], 'complex128');
    expect(np.real(z).toString()).toBe('[1. 3.]');
    expect(np.imag(z).toString()).toBe('[2. 4.]');
  });

  it('prints bigint views', () => {
    const a = np.array([1n, 2n, 3n, 4n], 'int64').reshape([2, 2]).T;
    expect(a.toString()).toBe('[[1 3]\n [2 4]]');
  });

  it('summarizes a large view from the right elements', () => {
    const a = np.arange(2000).slice('::-1');
    expect(a.toString()).toBe('[1.999e+03 1.998e+03 1.997e+03 ... 2.000e+00 1.000e+00 0.000e+00]');
  });

  it('prints zero-size views', () => {
    expect(np.zeros([0, 3]).T.toString()).toBe('[]');
    expect(np.array_repr(np.zeros([0, 3], 'float32'))).toBe(
      'array([], shape=(0, 3), dtype=float32)',
    );
    expect(np.array_repr(np.zeros([0]))).toBe('array([], dtype=float64)');
  });
});

describe('array_repr layout', () => {
  it('separates rows with commas and aligns continuation lines under the bracket', () => {
    const a = np.arange(6, undefined, undefined, 'int8').reshape([2, 3]);
    expect(np.array_repr(a)).toBe('array([[0, 1, 2],\n       [3, 4, 5]], dtype=int8)');
  });

  it('indents 3-D blocks and separates them with blank lines', () => {
    const a = np.arange(8, undefined, undefined, 'int32').reshape([2, 2, 2]);
    expect(np.array_repr(a)).toBe(
      'array([[[0, 1],\n        [2, 3]],\n\n       [[4, 5],\n        [6, 7]]], dtype=int32)',
    );
  });

  it('omits the dtype for float64, int64, complex128 and bool', () => {
    expect(np.array_repr(np.array([1.5]))).toBe('array([1.5])');
    expect(np.array_repr(np.array([1n], 'int64'))).toBe('array([1])');
    expect(np.array_repr(np.array([true, false], 'bool'))).toBe('array([ True, False])');
    expect(np.array_repr(np.array([new np.Complex(1, 2)], 'complex128'))).toBe('array([1.+2.j])');
  });

  it('writes the dtype without quotes', () => {
    expect(np.array_repr(np.array([1.5, 2.5], 'float32'))).toBe('array([1.5, 2.5], dtype=float32)');
  });

  it('moves the dtype to its own line when it would pass the line width', () => {
    expect(np.array_repr(np.arange(20, undefined, undefined, 'float32'), 60)).toBe(
      'array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,\n       10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],\n      dtype=float32)',
    );
  });

  it('shows the shape of arrays above the threshold', () => {
    np.set_printoptions({ threshold: 5 });
    expect(np.array_repr(np.arange(10, undefined, undefined, 'int8'))).toBe(
      'array([0, 1, 2, ..., 7, 8, 9], shape=(10,), dtype=int8)',
    );
  });

  it('prints a 0-d array as its scalar', () => {
    expect(np.array_repr(np.array(true, 'bool'))).toBe('array(True)');
    expect(np.array_str(np.array(true, 'bool'))).toBe('True');
    expect(np.array_str(np.array(1e22))).toBe('1e+22');
    expect(np.array_str(np.array(0.1, 'float32'))).toBe('0.1');
    expect(np.array_str(np.array(new np.Complex(0, 2), 'complex128'))).toBe('2j');
    expect(np.array_str(np.array(new np.Complex(1, -2), 'complex64'))).toBe('(1-2j)');
    expect(np.array_str(np.array(2000, 'float16'))).toBe('2e+03');
    expect(np.array_str(np.array(7, 'int8'))).toBe('7');
  });
});

describe('array2string', () => {
  it('uses the prefix for indentation and width only', () => {
    const a = np.arange(4);
    expect(np.array2string(a, { prefix: 'abc(' })).toBe('[0. 1. 2. 3.]');
    const wide = np.arange(30);
    expect(np.array2string(wide, { prefix: 'abc(', max_line_width: 30 })).toBe(
      '[ 0.  1.  2.  3.  4.  5.\n      6.  7.  8.  9. 10. 11.\n     12. 13. 14. 15. 16. 17.\n     18. 19. 20. 21. 22. 23.\n     24. 25. 26. 27. 28. 29.]',
    );
  });

  it('honours sign for integers and floats', () => {
    expect(np.array2string(np.array([1, -2, 3], 'int32'), { sign: '+' })).toBe('[+1 -2 +3]');
    expect(np.array2string(np.array([1, 2, 3], 'int32'), { sign: ' ' })).toBe('[ 1  2  3]');
    expect(np.array2string(np.array([-1, 2, 3], 'int32'), { sign: ' ' })).toBe('[-1  2  3]');
    expect(np.array2string(np.array([1.5, -2.5]), { sign: '+' })).toBe('[+1.5 -2.5]');
    expect(np.array2string(np.array([1.5, 2.5]), { sign: ' ' })).toBe('[ 1.5  2.5]');
  });

  it('pads non-finite values to the column width', () => {
    const a = np.array([1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]);
    expect(np.array2string(a)).toBe('[ 1.5  nan  inf -inf]');
    expect(np.array2string(a, { sign: '+' })).toBe('[+1.5 +nan +inf -inf]');
  });

  it('suppress_small switches off the exponent form for tiny values', () => {
    const a = np.array([1e-5, 1]);
    expect(np.array2string(a)).toBe('[1.e-05 1.e+00]');
    expect(np.array2string(a, { suppress_small: true })).toBe('[0.00001 1.     ]');
  });

  it('keeps the default suppress option off', () => {
    expect(np.get_printoptions().suppress).toBe(false);
  });

  it('picks the exponent cutoff from the float width', () => {
    expect(np.array2string(np.array([100000, 1], 'float32'))).toBe('[1.e+05 1.e+00]');
    expect(np.array2string(np.array([1000, 1], 'float16'))).toBe('[1.e+03 1.e+00]');
    expect(np.array2string(np.array([100000, 99999], 'float32'))).toBe('[100000.  99999.]');
    expect(np.array2string(np.array([1000000, 999999], 'float32'))).toBe(
      '[1.00000e+06 9.99999e+05]',
    );
    expect(np.array2string(np.array([10000000, 9999999], 'float64'))).toBe('[10000000.  9999999.]');
  });
});

describe('format_float_positional and format_float_scientific', () => {
  it('lets unique digits win over precision', () => {
    expect(np.format_float_positional(3.14, 4)).toBe('3.14');
    expect(np.format_float_positional(1000, 2)).toBe('1000.');
    expect(np.format_float_positional(3.14, 4, false)).toBe('3.1400');
    expect(np.format_float_scientific(1500)).toBe('1.5e+03');
    expect(np.format_float_scientific(1500, 3)).toBe('1.5e+03');
    expect(np.format_float_scientific(1500, 3, false)).toBe('1.500e+03');
  });

  it('accepts a boolean sign', () => {
    expect(np.format_float_positional(3.14, null, true, true, 'k', true)).toBe('+3.14');
    expect(np.format_float_positional(3.14, null, true, true, 'k', false)).toBe('3.14');
    expect(np.format_float_scientific(1500, null, true, 'k', true)).toBe('+1.5e+03');
    expect(np.format_float_positional(-3.14, null, true, true, 'k', true)).toBe('-3.14');
  });

  it('keeps a leading space for positive values when sign is a space', () => {
    expect(np.format_float_positional(3.14, null, true, true, 'k', ' ')).toBe(' 3.14');
    expect(np.format_float_scientific(1500, null, true, 'k', ' ')).toBe(' 1.5e+03');
  });

  it('pads with spaces on the right, not zeros', () => {
    expect(np.format_float_positional(1.5, null, true, true, 'k', false, 6, 4)).toBe('     1.5   ');
  });

  it('rejects arguments NumPy rejects', () => {
    expect(() => np.format_float_positional(1.5, null, false)).toThrow(/precision/);
    expect(() => np.format_float_scientific(1.5, null, false)).toThrow(/precision/);
    expect(() => np.format_float_positional(1.5, 1, true, true, 'k', false, null, null, 2)).toThrow(
      /min_digits/,
    );
    expect(() => np.format_float_positional(1.5, 0, true, false)).toThrow(/precision/);
  });

  it('rounds exact ties to even', () => {
    expect(np.format_float_positional(0.125, 2)).toBe('0.12');
    expect(np.format_float_positional(2.5, 0)).toBe('2.');
    expect(np.format_float_positional(0.5, 0)).toBe('0.');
    expect(np.format_float_positional(1.5, 0)).toBe('2.');
  });

  it('carries through nines and into a new digit', () => {
    expect(np.format_float_positional(9.99, 1)).toBe('10.');
    expect(np.format_float_positional(99.99, 1, false)).toBe('100.0');
    expect(np.format_float_scientific(9.99, 1)).toBe('1.e+01');
    expect(np.format_float_positional(0.5, 0, false)).toBe('0.');
    expect(np.format_float_positional(0.9, 0, false)).toBe('1.');
    expect(np.format_float_positional(0.04, 0, false)).toBe('0.');
  });

  it('prints negative zero and non-finite values', () => {
    expect(np.format_float_positional(-0)).toBe('-0.');
    expect(np.format_float_positional(Number.NaN, null, true, true, 'k', true)).toBe('nan');
    expect(np.format_float_positional(Number.NEGATIVE_INFINITY)).toBe('-inf');
    expect(np.format_float_scientific(Number.POSITIVE_INFINITY, null, true, 'k', true)).toBe(
      '+inf',
    );
    expect(np.format_float_scientific(0)).toBe('0.e+00');
  });
});

describe('contiguity flags', () => {
  it('ignores axes of length one and treats empty arrays as contiguous', () => {
    const col = np.arange(6).reshape([6, 1]);
    expect(col.flags.C_CONTIGUOUS).toBe(true);
    expect(col.flags.F_CONTIGUOUS).toBe(true);
    const empty = np.zeros([0, 3]).T;
    expect(empty.flags.C_CONTIGUOUS).toBe(true);
    expect(empty.flags.F_CONTIGUOUS).toBe(true);
    const m = np.arange(12).reshape([3, 4]);
    expect(m.flags.C_CONTIGUOUS).toBe(true);
    expect(m.flags.F_CONTIGUOUS).toBe(false);
    expect(m.T.flags.C_CONTIGUOUS).toBe(false);
    expect(m.T.flags.F_CONTIGUOUS).toBe(true);
  });
});

describe('view ownership', () => {
  it('free-function reshape, ravel, atleast_3d and diagonal return views', () => {
    const a = np.arange(12).reshape([3, 4]);
    for (const v of [np.reshape(a, [4, 3]), np.ravel(a), np.atleast_3d(a), np.diagonal(a)]) {
      expect(v.base).not.toBeNull();
      expect(v.flags.OWNDATA).toBe(false);
    }
    const owner = np.zeros([3, 4]);
    expect(np.reshape(owner, [12]).base).toBe(owner);
    expect(np.diagonal(owner).base).toBe(owner);
  });

  it('writes through views reach the input', () => {
    const a = np.zeros([3, 3]);
    np.diagonal(a).set([1], 5);
    expect(a.get([1, 1])).toBe(5);
    np.diag(a).set([2], 7);
    expect(a.get([2, 2])).toBe(7);
    const b = np.zeros([2, 3]);
    np.reshape(b.T, [3, 2, 1]).set([1, 1, 0], 9);
    expect(b.get([1, 1])).toBe(9);
    np.flip(b).set([0, 0], 4);
    expect(b.get([1, 2])).toBe(4);
  });

  it('squeeze and atleast_nd return the input itself when nothing changes', () => {
    const m = np.arange(6).reshape([2, 3]);
    expect(np.squeeze(m)).toBe(m);
    expect(np.atleast_1d(m)).toBe(m);
    expect(np.atleast_2d(m)).toBe(m);
    const cube = np.arange(24).reshape([2, 3, 4]);
    expect(np.atleast_3d(cube)).toBe(cube);
    const [first, second] = np.atleast_2d(m, np.arange(3));
    expect(first).toBe(m);
    expect(second.shape).toEqual([1, 3]);
    expect(second.base).not.toBeNull();
  });

  it('squeeze with an axis and expand_dims stay views', () => {
    const a = np.zeros([1, 3]);
    expect(np.squeeze(a, 0).base).toBe(a);
    expect(np.expand_dims(a, 0).base).toBe(a);
  });

  it('real of a real array is the array, imag is a fresh zeros array', () => {
    const a = np.arange(4);
    expect(np.real(a)).toBe(a);
    expect(np.imag(a).base).toBeNull();
    const z = np.array([new np.Complex(1, 2)], 'complex128');
    expect(np.real(z).base).toBe(z);
    expect(np.imag(z).base).toBe(z);
  });

  it('unstack, flip, rot90 and broadcast_arrays return views', () => {
    const a = np.arange(6).reshape([2, 3]);
    expect(np.unstack(a)[0]!.base).toBe(a.base);
    expect(np.flip(a).base).toBe(a.base);
    expect(np.rot90(a).base).toBe(a.base);
    expect(np.rot90(a, 0).base).toBe(a.base);
    expect(np.rot90(a, 4).storage.data).toBe(a.storage.data);
    const owner = np.zeros([2, 3]);
    const [x, y] = np.broadcast_arrays(owner, np.zeros([4, 2, 3]));
    expect(x!.base).toBe(owner);
    expect(y!.base).toBeNull();
    const [p, q] = np.broadcast_arrays(owner, np.zeros([2, 3]));
    expect(p).toBe(owner);
    expect(q!.base).toBeNull();
  });

  it('diagonal is a view for every axis pair and offset', () => {
    const a = np.arange(24).reshape([2, 3, 4]);
    expect(np.diagonal(a, 0, 1, 2).toArray()).toEqual([
      [0, 5, 10],
      [12, 17, 22],
    ]);
    expect(np.diagonal(a, 1, 0, 2).toArray()).toEqual([
      [1, 14],
      [5, 18],
      [9, 22],
    ]);
    expect(np.diagonal(a, -1, 1, 2).toArray()).toEqual([
      [4, 9],
      [16, 21],
    ]);
    expect(np.diagonal(np.zeros([2, 2]), 5).shape).toEqual([0]);
    expect(np.diagonal(np.zeros([2, 2]), -5).shape).toEqual([0]);
  });

  it('diag reads strided vectors correctly', () => {
    const v = np.arange(10).slice('2:8:2');
    expect(np.diag(v).toArray()).toEqual([
      [2, 0, 0],
      [0, 4, 0],
      [0, 0, 6],
    ]);
    expect(np.diag(v, 1).shape).toEqual([4, 4]);
    expect(np.diag(v, -1).get([1, 0])).toBe(2);
    const z = np.diag(np.array([new np.Complex(1, 1), new np.Complex(2, -2)], 'complex128'));
    expect(z.get([1, 1])).toEqual(new np.Complex(2, -2));
    expect(np.diag(np.array([1n, 2n], 'int64')).get([1, 1])).toBe(2n);
    expect(() => np.diag(np.zeros([2, 2, 2]))).toThrow(/1-D or 2-D/);
  });

  it('diag of a matrix is the diagonal of the logical array', () => {
    const m = np.arange(12).reshape([3, 4]).T;
    expect(np.diag(m).toArray()).toEqual([0, 5, 10]);
    expect(np.diag(m, 1).toArray()).toEqual([4, 9]);
  });

  it('diagflat accepts sliced input', () => {
    const a = np.arange(24).reshape([4, 6]).slice('1:3', '0:2');
    const d = np.diagflat(a);
    expect(d.shape).toEqual([4, 4]);
    expect([0, 1, 2, 3].map((i) => d.get([i, i]))).toEqual([6, 7, 12, 13]);
  });

  it('reshape of a non-contiguous array copies when a view is impossible', () => {
    const a = np.arange(12).reshape([3, 4]);
    const r = np.reshape(a.T, [12]);
    expect(r.toArray()).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    expect(r.base).not.toBeNull();
    expect(r.base).not.toBe(a);
    expect([...r.base!.shape]).toEqual([4, 3]);
    r.set([0], 99);
    expect(a.get([0, 0])).toBe(0);
  });

  it('reshape of a complex or bigint non-contiguous array copies correctly', () => {
    const z = np
      .array(
        Array.from({ length: 6 }, (_, i) => new np.Complex(i, -i)),
        'complex128',
      )
      .reshape([2, 3]);
    expect(np.reshape(z.T, [6]).get([1])).toEqual(new np.Complex(3, -3));
    const b = np.array([1n, 2n, 3n, 4n], 'int64').reshape([2, 2]);
    expect(np.reshape(b.T, [4]).toArray()).toEqual([1n, 3n, 2n, 4n]);
  });

  it('method reshape and ravel follow the same ownership rules', () => {
    const a = np.arange(12).reshape([3, 4]);
    expect(a.reshape([12]).base).toBe(a.base);
    const hidden = a.T.reshape([12]);
    expect(hidden.base).not.toBeNull();
    expect(hidden.flags.OWNDATA).toBe(false);
    const raveled = a.T.ravel();
    expect(raveled.base).toBeNull();
    expect(raveled.toArray()).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
  });
});

describe('ascontiguousarray and asfortranarray', () => {
  it('return the input when it already has the layout', () => {
    const c = np.arange(6).reshape([2, 3]);
    expect(np.ascontiguousarray(c)).toBe(c);
    const f = c.T;
    expect(np.asfortranarray(f)).toBe(f);
  });

  it('copy into the requested layout otherwise', () => {
    const c = np.arange(6).reshape([2, 3]);
    const f = np.asfortranarray(c);
    expect(f).not.toBe(c);
    expect(f.flags.F_CONTIGUOUS).toBe(true);
    expect(f.flags.OWNDATA).toBe(true);
    expect(f.toArray()).toEqual(c.toArray());
    const back = np.ascontiguousarray(c.T);
    expect(back.flags.C_CONTIGUOUS).toBe(true);
    expect(back.toArray()).toEqual([
      [0, 3],
      [1, 4],
      [2, 5],
    ]);
  });

  it('promote 0-d input to one dimension and honour dtype', () => {
    const s = np.array(5);
    const c = np.ascontiguousarray(s);
    expect(c.shape).toEqual([1]);
    expect(c.base).toBe(s);
    expect(np.asfortranarray(s).shape).toEqual([1]);
    expect(np.ascontiguousarray(np.arange(3), 'int32').dtype).toBe('int32');
  });
});

describe('float16 values held in a wider buffer', () => {
  it('prints what a half-precision value would hold', () => {
    const wide = new ArrayStorage(
      new Float32Array([-70000, 0.1, 1e-9, 70000, -65519]),
      [5],
      [1],
      0,
      'float16',
    );
    expect(array2stringOps(wide)).toBe('[     -inf  1.00e-01  0.00e+00       inf -6.55e+04]');
  });
});

describe('reshape strides', () => {
  it('splits a strided axis into several without copying', () => {
    const a = np.arange(24).slice('::2');
    const r = np.reshape(a, [3, 4]);
    expect(r.storage.data).toBe(a.storage.data);
    expect(r.toArray()).toEqual([
      [0, 2, 4, 6],
      [8, 10, 12, 14],
      [16, 18, 20, 22],
    ]);
    r.set([1, 1], -1);
    expect(a.get([5])).toBe(-1);
  });

  it('keeps trailing axes of length one on a strided array', () => {
    const a = np.arange(10).slice('::2');
    const r = np.reshape(a, [5, 1, 1]);
    expect(r.storage.data).toBe(a.storage.data);
    expect(r.toArray()).toEqual([[[0]], [[2]], [[4]], [[6]], [[8]]]);
  });
});

describe('backward slices', () => {
  it('select nothing from empty arrays and clamp out-of-range starts like NumPy', () => {
    expect(np.zeros([0]).slice('::-1').shape).toEqual([0]);
    expect(np.zeros([0]).slice('5::-1').shape).toEqual([0]);
    expect(np.arange(5).slice('10::-1').toArray()).toEqual([4, 3, 2, 1, 0]);
    expect(np.arange(5).slice('-10::-1').toArray()).toEqual([]);
    expect(np.arange(5).slice('10:2:-1').toArray()).toEqual([4, 3]);
    expect(np.arange(5).slice('3::-1').toArray()).toEqual([3, 2, 1, 0]);
  });
});
