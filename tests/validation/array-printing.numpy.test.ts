/**
 * NumPy oracle for array printing: array2string, array_repr, array_str and NDArray.toString
 * on contiguous and non-contiguous views, every dtype family, print options and summarization,
 * plus the utility float formatters.
 */

import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

type Opts = Record<string, number | string | boolean>;
type Case = { name: string; js: () => ReturnType<typeof np.arange>; py: string };

const DTYPES = [
  'bool',
  'int8',
  'uint8',
  'int16',
  'int32',
  'uint32',
  'int64',
  'uint64',
  'float16',
  'float32',
  'float64',
  'complex64',
  'complex128',
] as const;
type Dt = (typeof DTYPES)[number];

const isComplex = (dt: Dt) => dt.startsWith('complex');
const isBig = (dt: Dt) => dt === 'int64' || dt === 'uint64';

/** The same n elements on both sides: 0..n-1 (every third true for bool, imag = re/2 for complex). */
function seq(dt: Dt, n: number): { js: () => ReturnType<typeof np.arange>; py: string } {
  if (dt === 'bool') {
    const vals = Array.from({ length: n }, (_, i) => i % 3 === 0);
    return {
      js: () => np.array(vals, 'bool') as never,
      py: `np.array([${vals.map((v) => (v ? 'True' : 'False')).join(',')}], dtype='bool')`,
    };
  }
  if (isComplex(dt)) {
    const vals = Array.from({ length: n }, (_, i) => new np.Complex(i, i / 2));
    return {
      js: () => np.array(vals, dt) as never,
      py: `(np.arange(${n}) * (1 + 0.5j)).astype('${dt}')`,
    };
  }
  if (isBig(dt)) {
    const vals = Array.from({ length: n }, (_, i) => BigInt(i));
    return { js: () => np.array(vals, dt) as never, py: `np.arange(${n}, dtype='${dt}')` };
  }
  return {
    js: () => np.arange(n, undefined, undefined, dt) as never,
    py: `np.arange(${n}, dtype='${dt}')`,
  };
}

type AnyArr = ReturnType<typeof np.arange>;

function viewCases(dt: Dt): Case[] {
  const s = (n: number) => seq(dt, n);
  const c = (name: string, n: number, jsFn: (a: AnyArr) => AnyArr, pyFn: string): Case => ({
    name,
    js: () => jsFn(s(n).js()),
    py: pyFn.replace('A', s(n).py),
  });
  return [
    c('1d', 7, (a) => a, 'A'),
    c('2d transpose', 6, (a) => a.reshape([2, 3]).T, '(A).reshape(2,3).T'),
    c('1d reversed', 10, (a) => a.slice('::-1'), '(A)[::-1]'),
    c('1d strided with offset', 10, (a) => a.slice('1::2'), '(A)[1::2]'),
    c(
      '2d reversed rows',
      12,
      (a) => a.reshape([3, 4]).slice('::-1', ':'),
      '(A).reshape(3,4)[::-1,:]',
    ),
    c(
      '2d offset window',
      12,
      (a) => a.reshape([3, 4]).slice('1:', '1:3'),
      '(A).reshape(3,4)[1:,1:3]',
    ),
    c('3d transpose', 24, (a) => a.reshape([2, 3, 4]).T, '(A).reshape(2,3,4).T'),
    c(
      '4d permuted',
      48,
      (a) => np.transpose(a.reshape([2, 3, 2, 4]), [2, 0, 3, 1]),
      'np.transpose((A).reshape(2,3,2,4),[2,0,3,1])',
    ),
    c('broadcast view', 3, (a) => np.broadcast_to(a, [2, 3]), 'np.broadcast_to(A,(2,3))'),
    c('0d', 1, (a) => a.reshape([]), '(A).reshape(())'),
    c('empty slice', 6, (a) => a.slice('0:0'), '(A)[0:0]'),
    c('empty reversed', 6, (a) => a.slice('0:0').slice('::-1'), '(A)[0:0][::-1]'),
    c('wide rows wrap', 60, (a) => a.reshape([3, 20]), '(A).reshape(3,20)'),
    c('wide rows wrap transposed', 60, (a) => a.reshape([3, 20]).T, '(A).reshape(3,20).T'),
    c('large 1d summarized', 2000, (a) => a, 'A'),
    c('large reversed summarized', 2000, (a) => a.slice('::-1'), '(A)[::-1]'),
    c('large 2d transpose summarized', 2400, (a) => a.reshape([40, 60]).T, '(A).reshape(40,60).T'),
    c(
      'large 3d strided summarized',
      3000,
      (a) => a.reshape([10, 10, 30]).slice('::-2', ':', '1::2'),
      '(A).reshape(10,10,30)[::-2,:,1::2]',
    ),
    c(
      'large 4d transpose summarized',
      5400,
      (a) => a.reshape([6, 5, 6, 30]).T,
      '(A).reshape(6,5,6,30).T',
    ),
  ];
}

function reference(py: string, opts: Opts = {}): string[] {
  const kw = Object.entries(opts)
    .map(
      ([k, v]) =>
        `${k}=${typeof v === 'string' ? `'${v}'` : v === true ? 'True' : v === false ? 'False' : v}`,
    )
    .join(', ');
  const code = `x = ${py}\nwith np.printoptions(${kw}):\n    result = [np.array2string(x), np.array_repr(x), np.array_str(x), str(x)]`;
  return runNumPy(code).value as string[];
}

function ours(x: AnyArr, opts: Opts = {}): string[] {
  np.set_printoptions({ ...DEFAULTS, ...opts } as never);
  return [np.array2string(x), np.array_repr(x), np.array_str(x), x.toString()];
}

const DEFAULTS = {
  precision: 8,
  threshold: 1000,
  edgeitems: 3,
  linewidth: 75,
  suppress: false,
  floatmode: 'maxprec',
  sign: '-',
  nanstr: 'nan',
  infstr: 'inf',
};

function expectSame(x: AnyArr, py: string, opts: Opts = {}) {
  const want = reference(py, opts);
  const got = ours(x, opts);
  const names = ['array2string', 'array_repr', 'array_str', 'toString'];
  for (let i = 0; i < names.length; i++) {
    expect(got[i], `${names[i]} ${JSON.stringify(opts)}`).toBe(want[i]);
  }
}

describe('NumPy Validation: array printing', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });
  afterEach(() => np.set_printoptions(DEFAULTS as never));

  describe('views of every dtype print like NumPy', () => {
    for (const dt of DTYPES) {
      describe(dt, () => {
        for (const c of viewCases(dt)) {
          it(c.name, () => expectSame(c.js(), c.py));
        }
      });
    }
  });

  describe('real and imag views of complex arrays', () => {
    for (const dt of ['complex64', 'complex128'] as const) {
      const z = seq(dt, 6);
      it(`np.real and np.imag of ${dt}`, () => {
        expectSame(np.real(z.js()), `np.real(${z.py})`);
        expectSame(np.imag(z.js()), `np.imag(${z.py})`);
      });
      it(`real and imag of a transposed ${dt} matrix`, () => {
        expectSame(np.real(z.js().reshape([2, 3]).T), `np.real((${z.py}).reshape(2,3).T)`);
        expectSame(
          np.imag(z.js().reshape([3, 2]).slice('::-1')),
          `np.imag((${z.py}).reshape(3,2)[::-1])`,
        );
      });
    }
  });

  describe('print options on views', () => {
    const optionSets: Opts[] = [
      { precision: 3 },
      { precision: 0 },
      { suppress: true },
      { linewidth: 30 },
      { linewidth: 20, precision: 2 },
      { threshold: 5, edgeitems: 2 },
      { threshold: 10, edgeitems: 1 },
      { floatmode: 'fixed' },
      { floatmode: 'unique' },
      { floatmode: 'maxprec_equal', precision: 4 },
      { sign: '+' },
      { sign: ' ' },
      { nanstr: 'NAN', infstr: 'INF' },
    ];
    const subjects: { name: string; js: () => AnyArr; py: string }[] = [
      {
        name: 'float64 transposed with mixed magnitudes',
        js: () =>
          np
            .array(
              [1.5, 1e-5, 123456.789, -2.5, 0, 3.14159265358979, 1e8, 0.1, 7, -0.0, 100, 2 / 3],
              'float64',
            )
            .reshape([3, 4]).T,
        py: 'np.array([1.5, 1e-5, 123456.789, -2.5, 0, 3.14159265358979, 1e8, 0.1, 7, -0.0, 100, 2/3]).reshape(3,4).T',
      },
      {
        name: 'float32 reversed with nan and inf',
        js: () =>
          np
            .array(
              [
                0.1,
                Number.NaN,
                2.5,
                Number.POSITIVE_INFINITY,
                -1,
                Number.NEGATIVE_INFINITY,
                1e6,
                3,
              ],
              'float32',
            )
            .slice('::-1'),
        py: "np.array([0.1, np.nan, 2.5, np.inf, -1, -np.inf, 1e6, 3], dtype='float32')[::-1]",
      },
      {
        name: 'float16 strided',
        js: () =>
          np.array([0.1, 100, 1000, 0.5, 65504, 1e-3, 2.5, 7, 0.3, 12], 'float16').slice('::2'),
        py: "np.array([0.1, 100, 1000, 0.5, 65504, 1e-3, 2.5, 7, 0.3, 12], dtype='float16')[::2]",
      },
      {
        name: 'int32 transposed with negatives',
        js: () => np.array([5, -12, 300, 7, -1, 0], 'int32').reshape([2, 3]).T,
        py: "np.array([5, -12, 300, 7, -1, 0], dtype='int32').reshape(2,3).T",
      },
      {
        name: 'complex128 window with nan',
        js: () =>
          np
            .array(
              [
                new np.Complex(1.5, -2),
                new np.Complex(Number.NaN, 1e-3),
                new np.Complex(0, Number.POSITIVE_INFINITY),
                new np.Complex(-7, 0.25),
                new np.Complex(100, 1),
                new np.Complex(0.1, 0.2),
              ],
              'complex128',
            )
            .reshape([2, 3])
            .slice(':', '1:'),
        py: 'np.array([1.5-2j, complex(np.nan,1e-3), complex(0,np.inf), -7+0.25j, 100+1j, 0.1+0.2j]).reshape(2,3)[:,1:]',
      },
      {
        name: 'float64 large transposed',
        js: () => np.arange(2400).reshape([40, 60]).T,
        py: 'np.arange(2400.0).reshape(40,60).T',
      },
      {
        name: 'float64 wide rows',
        js: () => np.linspace(0, 1, 60).reshape([3, 20]),
        py: 'np.linspace(0, 1, 60).reshape(3, 20)',
      },
    ];
    for (const sub of subjects) {
      for (const o of optionSets) {
        it(`${sub.name} with ${JSON.stringify(o)}`, () => expectSame(sub.js(), sub.py, o));
      }
    }
  });

  describe('values and layout', () => {
    const cases: [string, () => AnyArr, string][] = [
      [
        'float64 integers',
        () => np.array([1, 2, 3, 100, -5], 'float64'),
        'np.array([1,2,3,100,-5.])',
      ],
      [
        'float64 exponent spread',
        () => np.array([1e-10, 1, 1e10], 'float64'),
        'np.array([1e-10, 1, 1e10])',
      ],
      [
        'float64 repeating decimals',
        () => np.array([1 / 3, 2 / 3, 1], 'float64'),
        'np.array([1/3, 2/3, 1])',
      ],
      [
        'float64 negative zero',
        () => np.array([-0.0, 0.0, 1.5], 'float64'),
        'np.array([-0.0, 0.0, 1.5])',
      ],
      [
        'float64 all nan',
        () => np.array([Number.NaN, Number.NaN], 'float64'),
        'np.array([np.nan, np.nan])',
      ],
      [
        'float64 inf only',
        () => np.array([Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY], 'float64'),
        'np.array([np.inf, -np.inf])',
      ],
      [
        'float32 0.1',
        () => np.array([0.1, 0.25], 'float32'),
        "np.array([0.1, 0.25], dtype='float32')",
      ],
      [
        'float32 large',
        () => np.array([123456789, 1], 'float32'),
        "np.array([123456789, 1], dtype='float32')",
      ],
      [
        'float16 ratio rounds to 1000',
        () => np.array([100, 0.1], 'float16'),
        "np.array([100, 0.1], dtype='float16')",
      ],
      [
        'int64 wide',
        () => np.array([-9223372036854775808n, 5n], 'int64'),
        "np.array([-9223372036854775808, 5], dtype='int64')",
      ],
      [
        'uint64 wide',
        () => np.array([18446744073709551615n, 5n], 'uint64'),
        "np.array([18446744073709551615, 5], dtype='uint64')",
      ],
      ['bool all true', () => np.array([true, true], 'bool'), 'np.array([True, True])'],
      ['empty 1d', () => np.zeros([0], 'float64'), 'np.zeros(0)'],
      ['empty 2d float32', () => np.zeros([0, 3], 'float32'), "np.zeros((0,3), dtype='float32')"],
      ['empty 3d int8', () => np.zeros([2, 0, 3], 'int8'), "np.zeros((2,0,3), dtype='int8')"],
      ['0d float', () => np.array(3.14159265358979), 'np.array(3.14159265358979)'],
      ['0d float32', () => np.array(0.1, 'float32'), "np.array(0.1, dtype='float32')"],
      ['0d complex', () => np.array(new np.Complex(1.5, -2), 'complex128'), 'np.array(1.5-2j)'],
      ['0d complex zero real', () => np.array(new np.Complex(0, 2), 'complex128'), 'np.array(2j)'],
      ['0d bool', () => np.array(true, 'bool'), 'np.array(True)'],
      [
        '0d complex zero real negative imag',
        () => np.array(new np.Complex(0, -2), 'complex128'),
        'np.array(complex(0, -2))',
      ],
      ['0d false bool', () => np.array(false, 'bool'), 'np.array(False)'],
      [
        '0d complex negative zero real',
        () => np.array(new np.Complex(-0, 2), 'complex128'),
        'np.array(complex(-0.0, 2))',
      ],
      ['0d int8', () => np.array(7, 'int8'), "np.array(7, dtype='int8')"],
      ['0d float large', () => np.array(1e22), 'np.array(1e22)'],
      ['0d float small', () => np.array(1.5e-7), 'np.array(1.5e-7)'],
      ['0d float32 mid', () => np.array(1234567, 'float32'), "np.array(1234567, dtype='float32')"],
      ['0d float16', () => np.array(2000, 'float16'), "np.array(2000, dtype='float16')"],
    ];
    for (const [name, js, py] of cases) {
      it(name, () => expectSame(js(), py));
    }

    it('repr puts a long dtype suffix on its own line when it would pass the line width', () => {
      expectSame(
        np.arange(15, undefined, undefined, 'int8').reshape([3, 5]),
        "np.arange(15, dtype='int8').reshape(3,5)",
        {
          linewidth: 40,
        },
      );
      expectSame(np.arange(20, undefined, undefined, 'float32'), "np.arange(20, dtype='float32')", {
        linewidth: 60,
      });
    });

    it('repr appends the shape of arrays above the threshold', () => {
      expectSame(np.arange(30, undefined, undefined, 'int32'), "np.arange(30, dtype='int32')", {
        threshold: 10,
      });
    });
  });

  describe('array2string arguments', () => {
    const arr = () => np.arange(12).reshape([3, 4]).T;
    const py = 'np.arange(12.0).reshape(3,4).T';
    const run = (kw: string, opts: Record<string, unknown>) => {
      const want = runNumPy(`x = ${py}\nresult = np.array2string(x${kw})`).value;
      expect(np.array2string(arr(), opts as never)).toBe(want);
    };
    it('separator', () => run(", separator=', '", { separator: ', ' }));
    it('prefix reserves indent and width', () => run(", prefix='array('", { prefix: 'array(' }));
    it('suffix reserves width', () =>
      run(", suffix=')', max_line_width=12", { suffix: ')', max_line_width: 12 }));
    it('max_line_width', () => run(', max_line_width=20', { max_line_width: 20 }));
    it('precision and suppress_small', () =>
      run(', precision=2, suppress_small=True', { precision: 2, suppress_small: true }));
    it('threshold and edgeitems', () =>
      run(', threshold=4, edgeitems=1', { threshold: 4, edgeitems: 1 }));
    it('floatmode and sign', () =>
      run(", floatmode='fixed', sign='+'", { floatmode: 'fixed', sign: '+' }));
  });

  describe('format_float_positional and format_float_scientific', () => {
    const lit = (v: unknown) =>
      v === null
        ? 'None'
        : v === true
          ? 'True'
          : v === false
            ? 'False'
            : typeof v === 'string'
              ? `'${v}'`
              : String(v);

    const check = (kind: 'positional' | 'scientific', x: number, args: Record<string, unknown>) => {
      const pyArg = (v: number) =>
        Object.is(v, -0)
          ? '-0.0'
          : Number.isNaN(v)
            ? 'np.nan'
            : v === Infinity
              ? 'np.inf'
              : v === -Infinity
                ? '-np.inf'
                : v.toExponential(17).replace(/e\+?/, 'e');
      const kw = Object.entries(args)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${k}=${lit(v)}`)
        .join(', ');
      const want = runNumPy(
        `try:\n    result = np.format_float_${kind}(${pyArg(x)}${kw ? `, ${kw}` : ''})\nexcept Exception:\n    result = 'ERR'`,
      ).value;
      let got: string;
      try {
        const a = args as Record<string, never>;
        got =
          kind === 'positional'
            ? np.format_float_positional(
                x,
                a.precision ?? null,
                a.unique ?? true,
                a.fractional ?? true,
                a.trim ?? 'k',
                a.sign ?? false,
                a.pad_left ?? null,
                a.pad_right ?? null,
                a.min_digits ?? null,
              )
            : np.format_float_scientific(
                x,
                a.precision ?? null,
                a.unique ?? true,
                a.trim ?? 'k',
                a.sign ?? false,
                a.pad_left ?? null,
                a.exp_digits ?? null,
                a.min_digits ?? null,
              );
      } catch {
        got = 'ERR';
      }
      expect(got, `${kind} ${x} ${kw}`).toBe(want);
    };

    it('unique digits win over precision and never pad zeros', () => {
      check('positional', 3.14, { precision: 4 });
      check('positional', 1000, { precision: 2 });
      check('positional', 3.14159265, { precision: 4 });
      check('positional', 0.30000000000000004, { precision: 3 });
      check('scientific', 1500, {});
      check('scientific', 1500, { precision: 3 });
      check('scientific', 3.14159265, { precision: 4 });
    });

    it('unique=false prints exactly precision digits', () => {
      check('positional', 3.14, { precision: 4, unique: false });
      check('positional', 0.125, { precision: 2, unique: false });
      check('positional', 2.5, { precision: 0, unique: false });
      check('positional', 1234.5678, { precision: 3, unique: false, fractional: false });
      check('positional', 0.045, { precision: 3, unique: false, fractional: false });
      check('scientific', 1500, { precision: 3, unique: false });
      check('scientific', 0.1, { precision: 7, unique: false, trim: '-' });
    });

    it('requires precision when unique is false', () => {
      check('positional', 1.5, { unique: false });
      check('scientific', 1.5, { unique: false });
      check('positional', 1.5, { precision: 0, fractional: false });
      check('positional', 3.14159, { precision: 1, min_digits: 2 });
    });

    it('sign accepts booleans', () => {
      check('positional', 3.14, { sign: true });
      check('positional', -3.14, { sign: true });
      check('positional', 0, { sign: true });
      check('positional', Number.POSITIVE_INFINITY, { sign: true });
      check('scientific', 1500, { sign: true });
      check('scientific', 1500, { sign: false });
      check('scientific', -1.5e-10, { sign: true, exp_digits: 3, pad_left: 8 });
    });

    it('trim, padding and minimum digits', () => {
      for (const trim of ['k', '.', '0', '-']) {
        check('positional', 2, { trim });
        check('positional', 1.5, { trim, pad_left: 6, pad_right: 4 });
        check('scientific', 2, { trim });
        check('scientific', 1.5, { trim, min_digits: 4 });
      }
      check('positional', 1.5, { min_digits: 4 });
      check('positional', 0.1, { min_digits: 20 });
      check('scientific', 1.5, { min_digits: 5 });
      check('positional', 0, { pad_right: 5 });
    });

    it('min_digits counts digits from the first nonzero integer digit when fractional is false', () => {
      check('positional', 0.125, { fractional: false, min_digits: 6 });
      check('positional', 9.99, { fractional: false, min_digits: 3 });
      check('positional', 123.456, { fractional: false, min_digits: 8 });
      check('positional', 0.0125, { fractional: false, min_digits: 6 });
    });

    it('prints no exponent digits for a zero exponent when exp_digits is 0', () => {
      check('scientific', 1.5, { exp_digits: 0 });
      check('scientific', 15, { exp_digits: 0 });
    });

    it('non-finite values', () => {
      for (const v of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
        check('positional', v, {});
        check('scientific', v, {});
      }
    });

    it('matches NumPy over seeded random arguments', () => {
      let seed = 20240607;
      const rnd = () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const pick = <T>(a: T[]): T => a[Math.floor(rnd() * a.length)]!;
      const pool = [
        0,
        1,
        1.5,
        3.14,
        0.1,
        0.125,
        0.5,
        2.5,
        1000,
        1500,
        12345.678,
        1e-5,
        1.5e-7,
        1e16,
        1e20,
        123456789.123,
        0.30000000000000004,
        9.99,
        99.99,
        9.995,
        0.0001,
        5e-324,
        2 / 3,
        Math.PI,
        -2.675,
        0.045,
        1.005,
      ];
      for (let it = 0; it < 600; it++) {
        const x =
          rnd() < 0.8
            ? pick(pool) * (rnd() < 0.3 ? -1 : 1)
            : (rnd() - 0.5) * 10 ** Math.floor(rnd() * 20 - 8);
        const unique = rnd() < 0.6;
        const args: Record<string, unknown> = {
          unique,
          precision: unique && rnd() < 0.4 ? null : Math.floor(rnd() * 10),
          trim: pick(['k', '.', '0', '-']),
          sign: rnd() < 0.5,
        };
        if (rnd() < 0.3) args.pad_left = Math.floor(rnd() * 12);
        if (unique && rnd() < 0.3) args.min_digits = Math.floor(rnd() * 8);
        if (rnd() < 0.4) {
          if (rnd() < 0.3) args.exp_digits = 1 + Math.floor(rnd() * 4);
          check('scientific', x, args);
        } else {
          args.fractional = rnd() < 0.8;
          // min_digits with significant-digit precision is not exercised
          if (args.fractional === false) args.min_digits = undefined;
          if (rnd() < 0.3) args.pad_right = Math.floor(rnd() * 12);
          check('positional', x, args);
        }
      }
    });
  });

  describe('randomized arrays', () => {
    it('matches NumPy for seeded random shapes, dtypes, views and options', () => {
      let seed = 987654;
      const rnd = () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const pick = <T>(a: T[]): T => a[Math.floor(rnd() * a.length)]!;
      const pool = [
        0,
        1,
        -1,
        1.5,
        3.14159265358979,
        0.1,
        0.125,
        1e-5,
        1e5,
        123456.789,
        1e8,
        0.30000000000000004,
        Number.NaN,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        100,
        -2.5,
        7,
        1000,
        2 / 3,
        -0,
      ];
      const pyNum = (v: number) =>
        Object.is(v, -0)
          ? '-0.0'
          : Number.isNaN(v)
            ? 'np.nan'
            : v === Infinity
              ? 'np.inf'
              : v === -Infinity
                ? '-np.inf'
                : v.toExponential(17).replace(/e\+?/, 'e');
      const optionSets: Opts[] = [
        {},
        { precision: 3 },
        { suppress: true },
        { linewidth: 30 },
        { threshold: 5, edgeitems: 2 },
        { floatmode: 'fixed' },
        { floatmode: 'unique' },
        { floatmode: 'maxprec_equal', precision: 4 },
        { sign: '+' },
        { sign: ' ' },
      ];
      const dts = [
        'float64',
        'float32',
        'float16',
        'complex128',
        'complex64',
        'int32',
        'int64',
        'int8',
        'uint8',
        'bool',
      ] as const;
      for (let it = 0; it < 120; it++) {
        const dt = pick([...dts]);
        const rank = Math.floor(rnd() * 5);
        const shape = Array.from(
          { length: rank },
          () => 1 + Math.floor(rnd() * (rank > 2 ? 4 : 9)),
        );
        if (rank > 0 && rnd() < 0.15) shape[Math.floor(rnd() * rank)] = rnd() < 0.5 ? 0 : 25;
        const size = shape.reduce((a, b) => a * b, 1);
        const vals: unknown[] = [];
        const py: string[] = [];
        for (let i = 0; i < size; i++) {
          if (dt.startsWith('complex')) {
            const re = pick(pool);
            const im = pick(pool);
            vals.push(new np.Complex(re, im));
            py.push(`complex(${pyNum(re)},${pyNum(im)})`);
          } else if (dt.startsWith('float')) {
            const v = pick(pool);
            vals.push(v);
            py.push(pyNum(v));
          } else if (dt === 'bool') {
            const v = rnd() < 0.5;
            vals.push(v);
            py.push(v ? 'True' : 'False');
          } else {
            const lim = dt === 'int8' ? 100 : dt === 'uint8' ? 250 : 100000;
            const v = dt === 'uint8' ? Math.floor(rnd() * lim) : Math.floor((rnd() - 0.4) * lim);
            vals.push(dt === 'int64' ? BigInt(v) : v);
            py.push(String(v));
          }
        }
        const transpose = rank >= 2 && rnd() < 0.5;
        const reverse = rank >= 1 && rnd() < 0.3;
        let x =
          size === 0 ? np.zeros(shape, dt) : (np.array(vals as never, dt).reshape(shape) as AnyArr);
        if (transpose) x = np.transpose(x);
        if (reverse) x = x.slice('::-1');
        const build = `np.array([${py.join(',')}], dtype='${dt}').reshape(${JSON.stringify(shape)})${transpose ? '.T' : ''}${reverse ? '[::-1]' : ''}`;
        expectSame(
          x,
          size === 0
            ? `np.zeros(${JSON.stringify(shape)}, dtype='${dt}')${transpose ? '.T' : ''}${reverse ? '[::-1]' : ''}`
            : build,
          pick(optionSets),
        );
      }
    });
  });
});
