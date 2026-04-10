/**
 * DType Sweep: Binary dtype cross-promotion (ALL × ALL).
 * Tests binary ops with every cross-dtype pair (14×13 = 182 combinations).
 * Validates that result dtype and values match NumPy's promotion rules.
 *
 * Ops selected to cover every distinct code path:
 * - add/subtract/multiply: elementwiseBinaryOp + custom fast paths
 * - divide: custom float promotion path
 * - power: custom bool promotion + complex power
 * - maximum/minimum: complexMinMax + WASM max/min
 * - mod/floor_divide: real-only elementwise
 * - fmax/fmin: complexMinMax (ignoreNaN variant)
 * - heaviside: custom WASM path
 * - float_power: always promotes to float64
 * - gcd: integer-only
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  expectBothRejectPre,
  expectMatchPre,
  isComplex,
  isInt,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

/** All cross-dtype pairs (same-dtype covered by binary-arithmetic sweep). */
const DTYPE_PAIRS: [string, string][] = [];
for (const dtA of ALL_DTYPES) {
  for (const dtB of ALL_DTYPES) {
    if (dtA === dtB) continue;
    DTYPE_PAIRS.push([dtA, dtB]);
  }
}

/** 14 ops covering every distinct code path. */
const BINARY_OPS: { name: string; fn: (a: any, b: any) => any }[] = [
  { name: 'add', fn: np.add },
  { name: 'subtract', fn: np.subtract },
  { name: 'multiply', fn: np.multiply },
  { name: 'divide', fn: np.divide },
  { name: 'power', fn: np.power },
  { name: 'maximum', fn: np.maximum },
  { name: 'minimum', fn: np.minimum },
  { name: 'mod', fn: np.mod },
  { name: 'floor_divide', fn: np.floor_divide },
  { name: 'fmax', fn: np.fmax },
  { name: 'fmin', fn: np.fmin },
  { name: 'float_power', fn: np.float_power },
  { name: 'heaviside', fn: np.heaviside },
  { name: 'gcd', fn: np.gcd },
];

// Functions that only accept real dtypes — complex rejected by both JS and NumPy
const REAL_ONLY = new Set([
  'maximum',
  'minimum',
  'mod',
  'floor_divide',
  'fmax',
  'fmin',
  'heaviside',
]);
// Functions that only accept integer dtypes
const INTEGER_ONLY = new Set(['gcd']);
// NumPy rejects boolean subtract
const NO_BOOL = new Set(['subtract']);

function dataForDtype(dtype: string): number[] {
  return dtype === 'bool' ? [1, 1, 1, 1] : [6, 7, 8, 9];
}

function dataForDtype2(dtype: string): number[] {
  return dtype === 'bool' ? [1, 1, 1, 1] : [1, 2, 3, 4];
}

function shouldReject(name: string, dtA: string, dtB: string): { reject: boolean; reason: string } {
  if (REAL_ONLY.has(name) && (isComplex(dtA) || isComplex(dtB))) {
    return { reject: true, reason: `${name} is not defined for complex numbers` };
  }
  if (INTEGER_ONLY.has(name) && (!isInt(dtA) || !isInt(dtB))) {
    return { reject: true, reason: `${name} is only defined for integer dtypes` };
  }
  if (NO_BOOL.has(name) && dtA === 'bool' && dtB === 'bool') {
    return { reject: true, reason: `NumPy disallows boolean ${name}` };
  }
  return { reject: false, reason: '' };
}

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const { name } of BINARY_OPS) {
    for (const [dtA, dtB] of DTYPE_PAIRS) {
      const dataA = dataForDtype(dtA);
      const dataB = dataForDtype2(dtB);
      const promotedComplex = isComplex(dtA) || isComplex(dtB);
      const ac = promotedComplex ? 'np.complex128' : 'np.float64';
      snippets[`${name}_${dtA}_${dtB}`] = `
a = np.array(${JSON.stringify(dataA)}, dtype=${npDtype(dtA)})
b = np.array(${JSON.stringify(dataB)}, dtype=${npDtype(dtB)})
_result_orig = np.${name}(a, b)
result = _result_orig.astype(${ac})`;
    }
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Binary cross-promotion (ALL × ALL)', () => {
  for (const { name, fn } of BINARY_OPS) {
    describe(name, () => {
      for (const [dtA, dtB] of DTYPE_PAIRS) {
        it(`${dtA} × ${dtB}`, () => {
          const dataA = dataForDtype(dtA);
          const dataB = dataForDtype2(dtB);
          const key = `${name}_${dtA}_${dtB}`;
          const pyResult = oracle.get(key)!;

          const { reject, reason } = shouldReject(name, dtA, dtB);
          if (reject || pyResult.error) {
            const r = expectBothRejectPre(
              reason || pyResult.error || 'NumPy rejects this combo',
              () => fn(array(dataA, dtA), array(dataB, dtB)),
              pyResult
            );
            if (r === 'both-reject') return;
          }

          const a = array(dataA, dtA);
          const b = array(dataB, dtB);
          const jsResult = fn(a, b);
          expectMatchPre(jsResult, pyResult, { rtol: 1e-3 });
        });
      }
    });
  }
});
