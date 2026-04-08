/**
 * DType Sweep: Divmod function.
 * Tests across ALL dtypes, validated against NumPy.
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

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0] : [7, 8, 9];
    const data2 = dtype === 'bool' ? [1, 1, 1] : [2, 3, 4];
    snippets[`divmod_${dtype}`] = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
q, r = np.divmod(a, b)
_result_orig = q
result = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Divmod', () => {
  for (const dtype of ALL_DTYPES) {
    it(`divmod ${dtype}`, () => {
      const data1 = dtype === 'bool' ? [1, 1, 0] : [7, 8, 9];
      const data2 = dtype === 'bool' ? [1, 1, 1] : [2, 3, 4];
      const pyCode = `
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
q, r = np.divmod(a, b)
_result_orig = q
result = _result_orig.astype(np.float64)`;

      // complex: divmod is not defined for complex numbers (both reject)
      if (dtype.startsWith('complex')) {
        const r = expectBothReject(
          'divmod is not defined for complex numbers',
          () => np.divmod(array(data1, dtype), array(data2, dtype)),
          pyCode
        );
        if (r === 'both-reject') return;
      }

      const [q] = np.divmod(array(data1, dtype), array(data2, dtype)) as [any, any];
      expectMatchPre(q, oracle.get(`divmod_${dtype}`)!, { rtol: 1e-3 });
    });
  }
});
