/**
 * Validates the result-dtype SSOT helpers (src/common/dtype.ts) against real
 * NumPy. These helpers back the compile-time type tables (src/common/dtype-promotion.ts),
 * so proving them correct here means the generated types reflect real behavior:
 *
 *   helper(dtype)  ===  NumPy's actual output dtype
 *
 * This is the correctness backbone for the "type everything" result rules.
 */

import { describe, expect, it } from 'vitest';
import { argmax, argmin, argsort, array, flatnonzero, nanargmax } from '../../src';
import {
  absResultDtype,
  angleResultDtype,
  boolArithmeticDtype,
  DTYPES,
  type DType,
  mathResultDtype,
  mathResultDtypeCanonical,
  promoteDTypes,
  reductionAccumDtype,
  roundResultDtype,
  stdVarResultDtype,
  trueDivideResultDtype,
} from '../../src/common/dtype';
import { runNumPy } from './numpy-oracle';

// Canonical dtype list (single source of truth in src/common/dtype.ts).
const ALL_DTYPES: readonly DType[] = DTYPES;

const arr = (d: DType) => `np.ones(2, dtype='${d}')`;

/** Run a NumPy op over an array of dtype `d` and return the result's dtype string. */
function numpyDtype(expr: (a: string) => string, d: DType): string {
  return runNumPy(`import numpy as np\nresult = ${expr(arr(d))}`).dtype;
}

// Each rule: the helper under test and the NumPy op whose output dtype it predicts.
const RULES: {
  name: string;
  helper: (d: DType) => DType;
  op: (a: string) => string;
  dtypes?: DType[]; // restrict when the op is undefined for some dtypes
}[] = [
  { name: 'mathResultDtype (sqrt)', helper: mathResultDtypeCanonical, op: (a) => `np.sqrt(${a})` },
  {
    name: 'trueDivideResultDtype (true_divide)',
    helper: trueDivideResultDtype,
    op: (a) => `np.true_divide(${a}, ${a})`,
  },
  { name: 'stdVarResultDtype (std)', helper: stdVarResultDtype, op: (a) => `np.std(${a})` },
  { name: 'absResultDtype (abs)', helper: absResultDtype, op: (a) => `np.abs(${a})` },
  { name: 'roundResultDtype (round)', helper: roundResultDtype, op: (a) => `np.round(${a})` },
  { name: 'reductionAccumDtype (sum)', helper: reductionAccumDtype, op: (a) => `np.sum(${a})` },
  {
    name: 'boolArithmeticDtype (power)',
    helper: boolArithmeticDtype,
    op: (a) => `np.power(${a}, ${a})`,
    // NumPy rejects negative integer powers only with negative exponents; ones() is safe.
  },
  {
    name: 'angleResultDtype (angle)',
    helper: angleResultDtype,
    op: (a) => `np.angle(${a})`,
  },
];

describe('result-dtype helpers match NumPy', () => {
  for (const rule of RULES) {
    describe(rule.name, () => {
      const dtypes = rule.dtypes ?? ALL_DTYPES;
      it.each(dtypes)('%s', (d) => {
        expect(rule.helper(d)).toBe(numpyDtype(rule.op, d));
      });
    });
  }
});

// ---- Free-function result-rule mappings (P1–P3) ----
// Confirms the *choice of rule* for each newly-typed free function matches real
// NumPy (the underlying helpers are validated above; this validates the mapping).

const PAIRS: [DType, DType][] = [
  ['int32', 'float32'],
  ['int8', 'int16'],
  ['float32', 'float64'],
  ['uint8', 'int8'],
  ['complex64', 'float32'],
  ['int64', 'int64'],
];

/** Binary free functions the generator maps to `promote` (result = promote(a,b)). */
const PROMOTE_BINARY: {
  name: string;
  op: (a: string, b: string) => string;
  pairs?: [DType, DType][];
}[] = [
  { name: 'maximum', op: (a, b) => `np.maximum(${a}, ${b})` },
  { name: 'minimum', op: (a, b) => `np.minimum(${a}, ${b})` },
  { name: 'fmax', op: (a, b) => `np.fmax(${a}, ${b})` },
  { name: 'fmin', op: (a, b) => `np.fmin(${a}, ${b})` },
  {
    name: 'fmod',
    op: (a, b) => `np.fmod(${a}, ${b})`,
    pairs: [
      ['int8', 'int16'],
      ['float32', 'float64'],
      ['int64', 'int64'],
    ],
  },
  {
    name: 'gcd',
    op: (a, b) => `np.gcd(${a}, ${b})`,
    pairs: [
      ['int8', 'int16'],
      ['uint8', 'int8'],
      ['int64', 'int64'],
    ],
  },
  {
    name: 'lcm',
    op: (a, b) => `np.lcm(${a}, ${b})`,
    pairs: [
      ['int8', 'int16'],
      ['uint8', 'int8'],
      ['int64', 'int64'],
    ],
  },
  { name: 'kron', op: (a, b) => `np.kron(${a}, ${b})` },
  { name: 'convolve', op: (a, b) => `np.convolve(${a}, ${b})` },
  { name: 'correlate', op: (a, b) => `np.correlate(${a}, ${b})` },
  { name: 'union1d', op: (a, b) => `np.union1d(${a}, ${b})` },
  { name: 'intersect1d', op: (a, b) => `np.intersect1d(${a}, ${b})` },
  { name: 'setxor1d', op: (a, b) => `np.setxor1d(${a}, ${b})` },
];

describe('free-function promote mappings match NumPy', () => {
  for (const fn of PROMOTE_BINARY) {
    describe(fn.name, () => {
      it.each(fn.pairs ?? PAIRS)('%s,%s', (da, db) => {
        const expected = runNumPy(`import numpy as np\nresult = ${fn.op(arr(da), arr(db))}`).dtype;
        expect(promoteDTypes(da, db)).toBe(expected);
      });
    });
  }
});

// setdiff1d preserves ar1's dtype regardless of ar2 (mapped to `preserve`).
describe('free-function preserve mapping (setdiff1d) matches NumPy', () => {
  it.each(ALL_DTYPES)('%s', (d) => {
    const out = runNumPy(
      `import numpy as np\nresult = np.setdiff1d(${arr(d)}, np.ones(2, dtype='float32'))`,
    ).dtype;
    expect(out).toBe(d);
  });
});

// ---- P4–P6: contraction / tuple / poly / previously-deferred mappings ----
// The underlying helpers (Promote, TrueDivide, ReductionAccum, MathResult, …) are
// validated above; these lock in the per-function *rule choice* against NumPy.

// Unary functions whose result dtype the given helper predicts from the input dtype.
const UNARY_MAPPED: {
  name: string;
  op: (a: string) => string;
  expected: (d: DType) => string;
  dtypes?: DType[];
}[] = [
  // sinc/i0/unwrap → true-divide rule
  { name: 'sinc', op: (a) => `np.sinc(${a})`, expected: trueDivideResultDtype },
  {
    name: 'i0',
    op: (a) => `np.i0(${a})`,
    expected: trueDivideResultDtype,
    dtypes: ['float64', 'float32', 'float16', 'int32', 'int8', 'uint8', 'bool'],
  },
  {
    name: 'unwrap',
    op: (a) => `np.unwrap(${a})`,
    expected: trueDivideResultDtype,
    dtypes: ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool'],
  },
  // vander/polyder → promote(D, int64)
  { name: 'vander', op: (a) => `np.vander(${a})`, expected: (d) => promoteDTypes(d, 'int64') },
  {
    name: 'polyder',
    op: (a) => `np.polyder(${a})`,
    expected: (d) => promoteDTypes(d, 'int64'),
  },
  // corrcoef/cov/polyint → promote(D, float64)
  {
    name: 'corrcoef',
    op: (a) => `np.corrcoef(np.vstack([${a}, ${a}]))`,
    expected: (d) => promoteDTypes(d, 'float64'),
  },
  {
    name: 'cov',
    op: (a) => `np.cov(np.vstack([${a}, ${a}]))`,
    expected: (d) => promoteDTypes(d, 'float64'),
  },
  {
    name: 'polyint',
    op: (a) => `np.polyint(${a})`,
    expected: (d) => promoteDTypes(d, 'float64'),
  },
  // trace → reduction accumulator (sum of diagonal)
  {
    name: 'trace',
    op: (a) => `np.trace(np.diag(${a}))`,
    expected: reductionAccumDtype,
  },
];

describe('free-function unary mappings match NumPy', () => {
  for (const rule of UNARY_MAPPED) {
    describe(rule.name, () => {
      it.each(rule.dtypes ?? ALL_DTYPES)('%s', (d) => {
        expect(rule.expected(d)).toBe(numpyDtype(rule.op, d));
      });
    });
  }
});

// dot/inner/vecdot → promote(a, b) (contraction preserves NumPy promotion).
describe('contraction promote mappings match NumPy', () => {
  for (const [name, op] of [
    ['dot', (a: string, b: string) => `np.dot(${a}, ${b})`],
    ['inner', (a: string, b: string) => `np.inner(${a}, ${b})`],
    ['vecdot', (a: string, b: string) => `np.vecdot(${a}, ${b})`],
  ] as const) {
    describe(name, () => {
      it.each(PAIRS)('%s,%s', (da, db) => {
        const out = runNumPy(
          `import numpy as np\nresult = ${op(`np.ones(3, dtype='${da}')`, `np.ones(3, dtype='${db}')`)}`,
        ).dtype;
        expect(promoteDTypes(da, db)).toBe(out);
      });
    });
  }
});

// float_power → promote, then minimum precision float64 / complex128.
describe('float_power mapping matches NumPy', () => {
  it.each(PAIRS)('%s,%s', (da, db) => {
    const out = runNumPy(
      `import numpy as np\nresult = np.float_power(np.ones(3, dtype='${da}'), np.ones(3, dtype='${db}'))`,
    ).dtype;
    expect(promoteDTypes(promoteDTypes(da, db), 'float64')).toBe(out);
  });
});

// ldexp(x1, x2) preserves x1's (float) dtype.
describe('ldexp mapping matches NumPy', () => {
  it.each(['float64', 'float32', 'float16'] as DType[])('%s', (d) => {
    const out = runNumPy(
      `import numpy as np\nresult = np.ldexp(np.ones(3, dtype='${d}'), np.ones(3, dtype='int32'))`,
    ).dtype;
    expect(out).toBe(d);
  });
});

// Tuple ufuncs: frexp → [MathResult, int32], modf → [MathResult, MathResult],
// divmod → [Power, Power], polydiv → [Divide, Divide].
describe('tuple ufunc mappings match NumPy', () => {
  it.each(['float64', 'float32', 'float16', 'int32', 'int8', 'uint8', 'bool'] as DType[])(
    'frexp %s',
    (d) => {
      const r = runNumPy(
        `import numpy as np\nm, e = np.frexp(np.ones(3, dtype='${d}'))\nresult = m`,
      ).dtype;
      const e = runNumPy(
        `import numpy as np\nm, e = np.frexp(np.ones(3, dtype='${d}'))\nresult = e`,
      ).dtype;
      expect(mathResultDtypeCanonical(d)).toBe(r);
      expect(e).toBe('int32');
    },
  );
  it.each(['float64', 'float32', 'float16', 'int32', 'int8'] as DType[])('modf %s', (d) => {
    const f = runNumPy(
      `import numpy as np\nf, i = np.modf(np.ones(3, dtype='${d}'))\nresult = f`,
    ).dtype;
    expect(mathResultDtypeCanonical(d)).toBe(f);
  });
  const divmodPairs: [DType, DType][] = [
    ['int32', 'float32'],
    ['int8', 'int16'],
    ['float32', 'float64'],
    ['uint8', 'int8'],
    ['int64', 'int64'],
  ];
  it.each(divmodPairs)('divmod %s,%s', (da, db) => {
    const q = runNumPy(
      `import numpy as np\nq, r = np.divmod(np.ones(3, dtype='${da}'), np.ones(3, dtype='${db}'))\nresult = q`,
    ).dtype;
    expect(boolArithmeticDtype(promoteDTypes(da, db))).toBe(q);
  });
});

// polyadd/polysub/polymul → promote; polydiv → true-divide of promote.
describe('poly binary mappings match NumPy', () => {
  const pairs: [DType, DType][] = [
    ['float32', 'float32'],
    ['float64', 'float32'],
    ['complex64', 'float32'],
  ];
  for (const name of ['polyadd', 'polysub', 'polymul']) {
    it.each(pairs)(`${name} %s,%s`, (da, db) => {
      const out = runNumPy(
        `import numpy as np\nresult = np.${name}(np.ones(3, dtype='${da}'), np.ones(3, dtype='${db}'))`,
      ).dtype;
      expect(promoteDTypes(da, db)).toBe(out);
    });
  }
  it.each(pairs)('polydiv %s,%s', (da, db) => {
    const out = runNumPy(
      `import numpy as np\nq, r = np.polydiv(np.ones(3, dtype='${da}'), np.ones(2, dtype='${db}'))\nresult = q`,
    ).dtype;
    expect(trueDivideResultDtype(promoteDTypes(da, db))).toBe(out);
  });
});

// ---- Index-family carveout: deliberate divergence from NumPy ----
// NumPy returns int64 (`intp`) for index results. numpy-ts intentionally does
// NOT match this: int64 is BigInt-backed, so int64 indices would make element
// access a `bigint` and break `arr[idx]`. Instead argmin/argmax/nanarg* return
// int32 and the sort/where family returns float64 — both `number`-yielding.
// This test PINS both sides: it confirms NumPy really is int64 (so we'd notice
// if that assumption changed) AND that numpy-ts holds its number-index convention.
describe('index family diverges from NumPy by design (number-yielding, not int64)', () => {
  const numpyIndexDtype = (expr: string) => runNumPy(`import numpy as np\nresult = ${expr}`).dtype;
  const data = () => array([5, 3, 8, 1, 7], 'float32');

  it('argmin/argmax/nanarg* → int32 here, int64 in NumPy', () => {
    expect(numpyIndexDtype("np.argmin(np.array([5,3,8], dtype='float32'))")).toBe('int64');
    // Reduce over an axis so the result is an array whose dtype we can read.
    const a2 = array(
      [
        [5, 3],
        [8, 1],
      ],
      'float32',
    );
    expect(argmin(a2, 0).dtype).toBe('int32');
    expect(argmax(a2, 0).dtype).toBe('int32');
    expect(nanargmax(a2, 0).dtype).toBe('int32');
  });

  it('argsort/flatnonzero → float64 here, int64 in NumPy', () => {
    expect(numpyIndexDtype("np.argsort(np.array([5,3,8], dtype='float32'))")).toBe('int64');
    expect(argsort(data()).dtype).toBe('float64');
    expect(flatnonzero(data()).dtype).toBe('float64');
  });
});
