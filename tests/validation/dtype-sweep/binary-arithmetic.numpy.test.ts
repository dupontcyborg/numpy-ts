/**
 * DType Sweep: Binary element-wise arithmetic functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  expectBothReject,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

const binaryOps: { name: string; fn: (a: any, b: any) => any }[] = [
  { name: 'add', fn: np.add },
  { name: 'subtract', fn: np.subtract },
  { name: 'multiply', fn: np.multiply },
  { name: 'divide', fn: np.divide },
  { name: 'power', fn: np.power },
  { name: 'maximum', fn: np.maximum },
  { name: 'minimum', fn: np.minimum },
  { name: 'mod', fn: np.mod },
  { name: 'floor_divide', fn: np.floor_divide },
  { name: 'copysign', fn: np.copysign },
  { name: 'hypot', fn: np.hypot },
  { name: 'arctan2', fn: np.arctan2 },
  { name: 'logaddexp', fn: np.logaddexp },
  { name: 'logaddexp2', fn: np.logaddexp2 },
  { name: 'fmax', fn: np.fmax },
  { name: 'fmin', fn: np.fmin },
  { name: 'remainder', fn: np.remainder },
  { name: 'float_power', fn: np.float_power },
  { name: 'heaviside', fn: np.heaviside },
  { name: 'gcd', fn: np.gcd },
  { name: 'lcm', fn: np.lcm },
];

// Functions that only accept real dtypes — complex rejected by both JS and NumPy
const REAL_ONLY = new Set([
  'maximum',
  'minimum',
  'mod',
  'floor_divide',
  'copysign',
  'hypot',
  'arctan2',
  'logaddexp',
  'logaddexp2',
  'fmax',
  'fmin',
  'remainder',
  'heaviside',
  'fmod',
]);
// Functions that only accept integer dtypes — float/complex/bool rejected by both
const INTEGER_ONLY = new Set(['gcd', 'lcm']);
// NumPy rejects boolean subtract: "boolean subtract not supported"
const NO_BOOL = new Set(['subtract']);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const { name } of binaryOps) {
    for (const dtype of ALL_DTYPES) {
      const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
      const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
      snippets[`${name}_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.${name}(a, b)
result = _result_orig.astype(np.float64)`;
    }
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Binary arithmetic', () => {
  for (const { name, fn } of binaryOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
          const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
          const pyCode = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.${name}(a, b)
result = _result_orig.astype(np.float64)`;

          // complex128/complex64: real-only ops throw TypeError in both JS and NumPy
          if (dtype.startsWith('complex') && REAL_ONLY.has(name)) {
            const r = expectBothReject(
              `${name} is not defined for complex numbers`,
              () => fn(array(data1, dtype), array(data2, dtype)),
              pyCode
            );
            if (r === 'both-reject') return;
          }
          // gcd/lcm: only defined for integers — float/complex/bool rejected by NumPy
          if (
            INTEGER_ONLY.has(name) &&
            (dtype.startsWith('float') || dtype.startsWith('complex') || dtype === 'bool')
          ) {
            const r = expectBothReject(
              `${name} is only defined for integer dtypes`,
              () => fn(array(data1, dtype), array(data2, dtype)),
              pyCode
            );
            if (r === 'both-reject') return;
          }
          // subtract(bool): NumPy explicitly disallows boolean subtraction
          if (NO_BOOL.has(name) && dtype === 'bool') {
            const r = expectBothReject(
              `NumPy disallows boolean ${name}`,
              () => fn(array(data1, dtype), array(data2, dtype)),
              pyCode
            );
            if (r === 'both-reject') return;
          }

          const a = array(data1, dtype);
          const b = array(data2, dtype);
          const jsResult = fn(a, b);
          expectMatchPre(jsResult, oracle.get(`${name}_${dtype}`)!, { rtol: 1e-3 });
        });
      }
    });
  }
});
