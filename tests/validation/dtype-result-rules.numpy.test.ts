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
import {
  absResultDtype,
  angleResultDtype,
  boolArithmeticDtype,
  type DType,
  mathResultDtype,
  reductionAccumDtype,
  stdVarResultDtype,
  trueDivideResultDtype,
} from '../../src/common/dtype';
import { runNumPy } from './numpy-oracle';

const ALL_DTYPES: DType[] = [
  'float64',
  'float32',
  'float16',
  'complex128',
  'complex64',
  'int64',
  'int32',
  'int16',
  'int8',
  'uint64',
  'uint32',
  'uint16',
  'uint8',
  'bool',
];

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
  { name: 'mathResultDtype (sqrt)', helper: mathResultDtype, op: (a) => `np.sqrt(${a})` },
  {
    name: 'trueDivideResultDtype (true_divide)',
    helper: trueDivideResultDtype,
    op: (a) => `np.true_divide(${a}, ${a})`,
  },
  { name: 'stdVarResultDtype (std)', helper: stdVarResultDtype, op: (a) => `np.std(${a})` },
  { name: 'absResultDtype (abs)', helper: absResultDtype, op: (a) => `np.abs(${a})` },
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
