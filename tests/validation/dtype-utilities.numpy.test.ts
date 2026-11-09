/**
 * NumPy validation: promote_types, result_type, can_cast, min_scalar_type, isdtype,
 * isscalar, mintypecode and typename, exhaustively over the 14 dtypes and their edge cases.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import {
  array,
  Complex,
  can_cast,
  isdtype,
  isscalar,
  min_scalar_type,
  mintypecode,
  promote_types,
  result_type,
  typename,
} from '../../src';
import type { DType } from '../../src/common/dtype';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

const DTYPES: DType[] = [
  'bool',
  'int8',
  'int16',
  'int32',
  'int64',
  'uint8',
  'uint16',
  'uint32',
  'uint64',
  'float16',
  'float32',
  'float64',
  'complex64',
  'complex128',
];
const PY_LIST = JSON.stringify(DTYPES);
const CASTINGS = ['no', 'equiv', 'safe', 'same_kind', 'unsafe'] as const;

// Deterministic generator so failures reproduce.
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

describe('NumPy Validation: dtype utilities', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('promote_types', () => {
    it('matches NumPy for every dtype pair', () => {
      const np_ = runNumPy(`
D = ${PY_LIST}
result = {a + '|' + b: str(np.promote_types(a, b)) for a in D for b in D}
`).value as Record<string, string>;
      for (const a of DTYPES) {
        for (const b of DTYPES) {
          expect(promote_types(a, b), `${a}+${b}`).toBe(np_[`${a}|${b}`]);
        }
      }
    });

    it('rejects names that are not dtypes', () => {
      expect(() => promote_types('foo' as DType, 'int8')).toThrow(TypeError);
      expect(() => promote_types('int8', 5 as unknown as DType)).toThrow(TypeError);
    });
  });

  describe('result_type', () => {
    it('matches NumPy for every dtype pair and triple', () => {
      const np_ = runNumPy(`
import itertools
D = ${PY_LIST}
result = {'|'.join(t): str(np.result_type(*t)) for n in (1, 2, 3) for t in itertools.product(D, repeat=n)}
`).value as Record<string, string>;
      for (const key of Object.keys(np_)) {
        expect(result_type(...(key.split('|') as DType[])), key).toBe(np_[key]);
      }
    });

    it('matches NumPy for longer random sequences', () => {
      const rnd = lcg(12345);
      const seqs: DType[][] = [];
      for (let i = 0; i < 400; i++) {
        const n = 4 + Math.floor(rnd() * 4);
        seqs.push(Array.from({ length: n }, () => DTYPES[Math.floor(rnd() * DTYPES.length)]!));
      }
      const np_ = runNumPy(`
seqs = ${JSON.stringify(seqs)}
result = [str(np.result_type(*s)) for s in seqs]
`).value as string[];
      seqs.forEach((s, i) => {
        expect(result_type(...s), s.join(',')).toBe(np_[i]);
      });
    });

    it('accepts arrays and dtype names together', () => {
      const a = array([1, 2], 'int8');
      const b = array([1, 2], 'uint16');
      const np_ = runNumPy(
        "result = str(np.result_type(np.array([1, 2], 'int8'), np.array([1, 2], 'uint16'), 'float32'))",
      );
      expect(result_type(a, b, 'float32')).toBe(np_.value);
    });

    it('throws without arguments like NumPy', () => {
      const np_ = runNumPy(`
try:
    np.result_type()
    result = 'no error'
except ValueError:
    result = 'ValueError'
`);
      expect(np_.value).toBe('ValueError');
      expect(() => result_type()).toThrow();
    });
  });

  describe('can_cast', () => {
    it('matches NumPy for every dtype pair and casting rule', () => {
      const np_ = runNumPy(`
D = ${PY_LIST}
result = {a + '|' + b + '|' + c: bool(np.can_cast(a, b, casting=c)) for a in D for b in D for c in ${JSON.stringify(CASTINGS)}}
`).value as Record<string, boolean>;
      for (const a of DTYPES) {
        for (const b of DTYPES) {
          for (const c of CASTINGS) {
            expect(can_cast(a, b, c), `${a}->${b} ${c}`).toBe(np_[`${a}|${b}|${c}`]);
          }
        }
      }
    });

    it('takes the dtype of an array source', () => {
      const np_ = runNumPy("result = bool(np.can_cast(np.dtype('uint8'), 'int8', 'safe'))");
      expect(can_cast(array([1], 'uint8'), 'int8')).toBe(np_.value);
    });

    it('rejects unknown dtypes and unknown casting rules', () => {
      expect(() => can_cast('foo' as DType, 'int8')).toThrow(TypeError);
      expect(() => can_cast('int8', 'foo' as DType)).toThrow(TypeError);
      expect(() => can_cast('int8', 'int16', 'bogus' as 'safe')).toThrow(/casting must be/);
    });
  });

  describe('min_scalar_type', () => {
    const ints: (number | bigint)[] = [
      0, 1, 127, 128, 255, 256, 65535, 65536, 4294967295, 4294967296, -1, -128, -129, -32768,
      -32769, -2147483648, -2147483649,
    ];
    const bigints: bigint[] = [
      2n ** 63n - 1n,
      2n ** 63n,
      2n ** 64n - 1n,
      -(2n ** 63n),
      -32769n,
      -2147483649n,
      0n,
      255n,
      -1n,
    ];

    it('matches NumPy for integers and booleans', () => {
      const all = [...ints.map(String), ...bigints.map(String)];
      const np_ = runNumPy(`
vals = [${all.join(', ')}]
result = [str(np.min_scalar_type(v)) for v in vals] + [str(np.min_scalar_type(True)), str(np.min_scalar_type(False))]
`).value as string[];
      const got = [...ints, ...bigints].map((v) => min_scalar_type(v));
      expect(got).toEqual(np_.slice(0, all.length));
      expect([min_scalar_type(true), min_scalar_type(false)]).toEqual(np_.slice(all.length));
    });

    it('matches NumPy for non-integer floats, infinities and NaN', () => {
      const floats = [
        0.5,
        1.5,
        3.14,
        64999.5,
        65000.5,
        10000000000.5,
        3.3e38 + 1e30,
        3.5e38 + 1e30,
        1e-300,
        5e-324,
        -0,
        -64999.5,
        -65000.5,
        -3.3e38 - 1e30,
        -3.5e38 - 1e30,
        Infinity,
        -Infinity,
        NaN,
      ];
      const py = floats.map((f) => {
        if (Number.isNaN(f)) return "float('nan')";
        if (!Number.isFinite(f)) return f > 0 ? "float('inf')" : "float('-inf')";
        return Object.is(f, -0) ? '-0.0' : f.toExponential();
      });
      const np_ = runNumPy(`
vals = [${py.join(', ')}]
result = [str(np.min_scalar_type(v)) for v in vals]
`).value as string[];
      floats.forEach((v, i) => {
        expect(min_scalar_type(v), String(v)).toBe(np_[i]);
      });
    });

    it('reads integer-valued numbers as integers', () => {
      expect(min_scalar_type(1e10)).toBe('uint64');
      expect(min_scalar_type(2 ** 64)).toBe('float32');
      expect(min_scalar_type(1e300)).toBe('float64');
    });

    it('throws for bigints outside the 64-bit range', () => {
      const np_ = runNumPy('result = str(np.min_scalar_type(2**64))');
      expect(np_.value).toBe('object');
      expect(() => min_scalar_type(2n ** 64n)).toThrow(RangeError);
      expect(() => min_scalar_type(-(2n ** 63n) - 1n)).toThrow(RangeError);
    });
  });

  describe('isdtype', () => {
    const KINDS = [
      'bool',
      'signed integer',
      'unsigned integer',
      'integral',
      'real floating',
      'complex floating',
      'numeric',
    ];

    it('matches NumPy for every dtype and kind name', () => {
      const np_ = runNumPy(`
D = ${PY_LIST}
K = ${JSON.stringify(KINDS)}
result = {d + '|' + k: bool(np.isdtype(np.dtype(d), k)) for d in D for k in K}
`).value as Record<string, boolean>;
      for (const d of DTYPES) {
        for (const k of KINDS) {
          expect(isdtype(d, k), `${d} ${k}`).toBe(np_[`${d}|${k}`]);
        }
      }
    });

    it('matches NumPy when the kind is a dtype', () => {
      const np_ = runNumPy(`
D = ${PY_LIST}
result = {a + '|' + b: bool(np.isdtype(np.dtype(a), np.dtype(b))) for a in D for b in D}
`).value as Record<string, boolean>;
      for (const a of DTYPES) {
        for (const b of DTYPES) {
          expect(isdtype(a, b), `${a} ${b}`).toBe(np_[`${a}|${b}`]);
        }
      }
    });

    it('matches NumPy for tuples of kinds and dtypes', () => {
      const combos: string[][] = [
        [],
        ['bool'],
        ['bool', 'integral'],
        ['real floating', 'complex floating'],
        ['float32', 'float64'],
        ['float32', 'integral'],
        ['signed integer', 'uint8'],
        ['numeric', 'bool'],
        ['int8', 'int16', 'unsigned integer'],
      ];
      const np_ = runNumPy(`
D = ${PY_LIST}
combos = ${JSON.stringify(combos)}
def conv(k):
    return k if k in ${JSON.stringify(KINDS)} else np.dtype(k)
result = [[bool(np.isdtype(np.dtype(d), tuple(conv(k) for k in c))) for d in D] for c in combos]
`).value as boolean[][];
      combos.forEach((c, i) => {
        DTYPES.forEach((d, j) => {
          expect(isdtype(d, c), `${d} ${c.join(',')}`).toBe(np_[i]![j]);
        });
      });
    });

    it('throws for unknown kind strings like NumPy', () => {
      const np_ = runNumPy(`
try:
    np.isdtype(np.float64, 'f')
    result = 'no error'
except ValueError:
    result = 'ValueError'
`);
      expect(np_.value).toBe('ValueError');
      expect(() => isdtype('float64', 'f')).toThrow(/not a known kind name/);
      expect(() => isdtype('float64', 'floating')).toThrow(/not a known kind name/);
      expect(() => isdtype('float64', ['integral', 'nope'])).toThrow(/not a known kind name/);
      expect(() => isdtype('float64', 'constructor')).toThrow(/not a known kind name/);
    });

    it('throws a TypeError for a non-dtype argument or non-string kind', () => {
      expect(() => isdtype('foo' as DType, 'numeric')).toThrow(TypeError);
      expect(() => isdtype(5 as unknown as DType, 'numeric')).toThrow(TypeError);
      expect(() => isdtype('float64', 5 as unknown as string)).toThrow(TypeError);
      expect(() => isdtype('float64', [['numeric']] as unknown as string[])).toThrow(TypeError);
      expect(() => isdtype('float64', null as unknown as string)).toThrow(TypeError);
    });
  });

  describe('isscalar', () => {
    it('matches NumPy for every scalar form', () => {
      const np_ = runNumPy(`
result = {
  'number': bool(np.isscalar(1.5)),
  'integer': bool(np.isscalar(5)),
  'bigint': bool(np.isscalar(5)),
  'boolean': bool(np.isscalar(True)),
  'complex': bool(np.isscalar(1 + 2j)),
  'string': bool(np.isscalar('a')),
  'array': bool(np.isscalar([1, 2])),
  'ndarray': bool(np.isscalar(np.array([1, 2]))),
  'zerod': bool(np.isscalar(np.array(5))),
  'none': bool(np.isscalar(None)),
  'dict': bool(np.isscalar({'a': 1})),
}
`).value as Record<string, boolean>;
      expect(isscalar(1.5)).toBe(np_.number);
      expect(isscalar(5)).toBe(np_.integer);
      expect(isscalar(5n)).toBe(np_.bigint);
      expect(isscalar(true)).toBe(np_.boolean);
      expect(isscalar(new Complex(1, 2))).toBe(np_.complex);
      expect(isscalar('a')).toBe(np_.string);
      expect(isscalar([1, 2])).toBe(np_.array);
      expect(isscalar(array([1, 2]))).toBe(np_.ndarray);
      expect(isscalar(array(5))).toBe(np_.zerod);
      expect(isscalar(null)).toBe(np_.none);
      expect(isscalar(undefined)).toBe(np_.none);
      expect(isscalar({ a: 1 })).toBe(np_.dict);
    });
  });

  describe('mintypecode', () => {
    const CODES = 'bBhHiIlLqQfdgFDG?eSUO';

    it('matches NumPy for every list of up to three type codes', () => {
      const lists: string[][] = [[]];
      for (const a of CODES) {
        lists.push([a]);
        for (const b of CODES) {
          lists.push([a, b]);
          for (const c of CODES) lists.push([a, b, c]);
        }
      }
      const np_ = runNumPy(`
lists = ${JSON.stringify(lists)}
result = [np.mintypecode(l) for l in lists]
`).value as string[];
      lists.forEach((l, i) => {
        expect(mintypecode(l), l.join(',')).toBe(np_[i]);
        expect(mintypecode(l.join('')), l.join('')).toBe(np_[i]);
      });
    });

    it('honours typeset and default, as a string or an array', () => {
      const rnd = lcg(777);
      const typesets = ['GDFgdf', 'fd', 'FD', 'Dd', 'f', 'dg', 'bBhH', 'G', '', 'iIlL', 'fdFD'];
      const cases: { chars: string[]; typeset: string | string[]; def: string }[] = [];
      for (let i = 0; i < 400; i++) {
        const n = Math.floor(rnd() * 5);
        const chars = Array.from({ length: n }, () => CODES[Math.floor(rnd() * CODES.length)]!);
        const ts = typesets[Math.floor(rnd() * typesets.length)]!;
        cases.push({
          chars,
          typeset: rnd() < 0.5 ? ts : ts.split(''),
          def: 'dfDG'[Math.floor(rnd() * 4)]!,
        });
      }
      const np_ = runNumPy(`
cases = ${JSON.stringify(cases)}
result = [np.mintypecode(c['chars'], c['typeset'], c['def']) for c in cases]
`).value as string[];
      cases.forEach((c, i) => {
        expect(mintypecode(c.chars, c.typeset, c.def), JSON.stringify(c)).toBe(np_[i]);
      });
    });

    it('matches the documented NumPy results', () => {
      const np_ = runNumPy(`
result = [
  np.mintypecode(['i', 'f']),
  np.mintypecode(['F', 'd']),
  np.mintypecode(['d', 'f', 'S']),
  np.mintypecode('abceh', default='G'),
  np.mintypecode(['bb'], 'fd'),
  np.mintypecode('bb', 'fd'),
]
`).value as string[];
      expect([
        mintypecode(['i', 'f']),
        mintypecode(['F', 'd']),
        mintypecode(['d', 'f', 'S']),
        mintypecode('abceh', 'GDFgdf', 'G'),
        mintypecode(['bb'], 'fd'),
        mintypecode('bb', 'fd'),
      ]).toEqual(np_);
    });

    it('treats an empty-string element as a unicode code like NumPy', () => {
      const np_ = runNumPy("result = np.mintypecode(['', 'f'])");
      expect(mintypecode(['', 'f'])).toBe(np_.value);
    });
  });

  describe('typename', () => {
    const CODES = [
      'S1',
      '?',
      'b',
      'B',
      'h',
      'H',
      'i',
      'I',
      'l',
      'L',
      'q',
      'Q',
      'f',
      'd',
      'g',
      'F',
      'D',
      'G',
      'S',
      'U',
      'V',
      'O',
    ];

    it('matches NumPy for every type code', () => {
      const np_ = runNumPy(`
result = [np.typename(c) for c in ${JSON.stringify(CODES)}]
`).value as string[];
      CODES.forEach((c, i) => {
        expect(typename(c), c).toBe(np_[i]);
      });
    });

    it('throws for codes NumPy rejects', () => {
      const unknown = ['e', 'c', 'x', '', 'dd', 'float32', 'int8', 'm', 'M', 'a', 'p', 'P'];
      const np_ = runNumPy(`
out = []
for c in ${JSON.stringify(unknown)}:
    try:
        np.typename(c)
        out.append('ok')
    except KeyError:
        out.append('KeyError')
result = out
`).value as string[];
      expect(np_.every((v) => v === 'KeyError')).toBe(true);
      for (const c of unknown) expect(() => typename(c), c).toThrow(/unknown type code/);
    });
  });
});
