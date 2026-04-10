/**
 * DType Sweep: Index utility functions.
 * Most return integer indices (no dtype iteration needed).
 * Functions taking array inputs are tested across ALL_DTYPES.
 */
import { describe, it, beforeAll } from 'vitest';
import { expect } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  expectMatchPre,
  expectBothRejectPre,
  arraysClose,
  toComparable,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  // --- Pure index functions (no dtype iteration) ---

  snippets['indices'] = `
result = np.array(np.indices((3,3))).astype(np.float64)`;

  snippets['ix_'] = `
result = np.array(np.ix_(np.arange(3))[0]).astype(np.float64)`;

  snippets['ravel_multi_index'] = `
result = np.array(np.ravel_multi_index([[1,0],[0,1]], (2,2))).astype(np.float64)`;

  snippets['unravel_index'] = `
result = np.array(np.unravel_index([3,5], (3,3))).astype(np.float64)`;

  snippets['diag_indices'] = `
result = np.array(np.diag_indices(3)).astype(np.float64)`;

  snippets['tril_indices'] = `
result = np.array(np.tril_indices(3)).astype(np.float64)`;

  snippets['triu_indices'] = `
result = np.array(np.triu_indices(3)).astype(np.float64)`;

  snippets['mask_indices'] = `
result = np.array(np.mask_indices(3, np.triu)).astype(np.float64)`;

  // --- Array-input functions across ALL_DTYPES ---

  for (const dtype of ALL_DTYPES) {
    const dt = npDtype(dtype);

    snippets[`diag_indices_from_${dtype}`] = `
a = np.ones((3,3), dtype=${dt})
result = np.array(np.diag_indices_from(a)).astype(np.float64)`;

    snippets[`tril_indices_from_${dtype}`] = `
a = np.ones((3,3), dtype=${dt})
result = np.array(np.tril_indices_from(a)).astype(np.float64)`;

    snippets[`triu_indices_from_${dtype}`] = `
a = np.ones((3,3), dtype=${dt})
result = np.array(np.triu_indices_from(a)).astype(np.float64)`;

    // ix_ with typed input
    snippets[`ix__${dtype}`] = `
a = np.arange(3).astype(${dt})
result = np.array(np.ix_(a)[0]).astype(np.float64)`;

    // may_share_memory / shares_memory with typed arrays
    snippets[`may_share_memory_${dtype}`] = `
a = np.ones(4, dtype=${dt})
result = [bool(np.may_share_memory(a, a))]`;

    snippets[`shares_memory_${dtype}`] = `
a = np.ones(4, dtype=${dt})
result = [bool(np.shares_memory(a, a))]`;

    // mask_indices doesn't vary by dtype but we verify it works
    snippets[`mask_indices_${dtype}`] = `
result = np.array(np.mask_indices(3, np.triu)).astype(np.float64)`;

    // indices with dtype param
    snippets[`indices_${dtype}`] = `
_result_orig = np.indices((3,3), dtype=${dt})
result = np.array(_result_orig).astype(np.float64)`;

    // ravel_multi_index with typed input
    snippets[`ravel_multi_index_${dtype}`] = `
a = np.array([[1,0],[0,1]], dtype=${dt})
result = np.array(np.ravel_multi_index(a, (2,2))).astype(np.float64)`;

    // unravel_index with typed input
    snippets[`unravel_index_${dtype}`] = `
a = np.array([3,5], dtype=${dt})
result = np.array(np.unravel_index(a, (3,3))).astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Index utilities (no dtype iteration)', () => {
  it('indices((3,3))', () => {
    const jsResult = np.indices([3, 3]);
    expectMatchPre(jsResult, oracle.get('indices')!, { rtol: 0, atol: 0, indexResult: true });
  });

  it('ix_(arange(3))', () => {
    const [first] = np.ix_(np.arange(3));
    const pyResult = oracle.get('ix_')!;
    if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
    expect(arraysClose(first.toArray(), pyResult.value, 0, 0)).toBe(true);
  });

  it('ravel_multi_index', () => {
    const jsResult = np.ravel_multi_index([array([1, 0]), array([0, 1])], [2, 2]);
    expectMatchPre(jsResult, oracle.get('ravel_multi_index')!, {
      rtol: 0,
      atol: 0,
      indexResult: true,
    });
  });

  it('unravel_index', () => {
    const tupleResult = np.unravel_index(array([3, 5]), [3, 3]);
    const jsResult = np.array(tupleResult.map((a: any) => toComparable(a)));
    expectMatchPre(jsResult, oracle.get('unravel_index')!, { rtol: 0, atol: 0, indexResult: true });
  });

  it('diag_indices(3)', () => {
    const tupleResult = np.diag_indices(3);
    const jsResult = np.array(tupleResult.map((a: any) => toComparable(a)));
    expectMatchPre(jsResult, oracle.get('diag_indices')!, { rtol: 0, atol: 0, indexResult: true });
  });

  it('tril_indices(3)', () => {
    const tupleResult = np.tril_indices(3);
    const jsResult = np.array(tupleResult.map((a: any) => toComparable(a)));
    expectMatchPre(jsResult, oracle.get('tril_indices')!, { rtol: 0, atol: 0, indexResult: true });
  });

  it('triu_indices(3)', () => {
    const tupleResult = np.triu_indices(3);
    const jsResult = np.array(tupleResult.map((a: any) => toComparable(a)));
    expectMatchPre(jsResult, oracle.get('triu_indices')!, { rtol: 0, atol: 0, indexResult: true });
  });

  it('mask_indices(3, np.triu)', () => {
    const tupleResult = np.mask_indices(3, np.triu);
    const jsResult = np.array(tupleResult.map((a: any) => toComparable(a)));
    expectMatchPre(jsResult, oracle.get('mask_indices')!, { rtol: 0, atol: 0, indexResult: true });
  });
});

describe('DType Sweep: diag_indices_from (across dtypes)', () => {
  for (const dtype of ALL_DTYPES) {
    it(dtype, () => {
      const a = np.ones([3, 3], dtype);
      const tupleResult = np.diag_indices_from(a);
      const jsResult = np.array(tupleResult.map((r: any) => toComparable(r)));
      expectMatchPre(jsResult, oracle.get(`diag_indices_from_${dtype}`)!, {
        rtol: 0,
        atol: 0,
        indexResult: true,
      });
    });
  }
});

describe('DType Sweep: tril_indices_from (across dtypes)', () => {
  for (const dtype of ALL_DTYPES) {
    it(dtype, () => {
      const a = np.ones([3, 3], dtype);
      const tupleResult = np.tril_indices_from(a);
      const jsResult = np.array(tupleResult.map((r: any) => toComparable(r)));
      expectMatchPre(jsResult, oracle.get(`tril_indices_from_${dtype}`)!, {
        rtol: 0,
        atol: 0,
        indexResult: true,
      });
    });
  }
});

describe('DType Sweep: triu_indices_from (across dtypes)', () => {
  for (const dtype of ALL_DTYPES) {
    it(dtype, () => {
      const a = np.ones([3, 3], dtype);
      const tupleResult = np.triu_indices_from(a);
      const jsResult = np.array(tupleResult.map((r: any) => toComparable(r)));
      expectMatchPre(jsResult, oracle.get(`triu_indices_from_${dtype}`)!, {
        rtol: 0,
        atol: 0,
        indexResult: true,
      });
    });
  }
});

describe('DType Sweep: ix_ (across dtypes)', () => {
  describe('ix_', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const a = np.arange(3).astype(dtype);
        const pyResult = oracle.get(`ix__${dtype}`)!;
        const r = expectBothRejectPre(`ix_ may not support ${dtype}`, () => np.ix_(a), pyResult);
        if (r === 'both-reject') return;
        const [first] = np.ix_(a);
        if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
        expect(arraysClose(first.toArray(), pyResult.value, 0, 0)).toBe(true);
      });
    }
  });
});

describe('DType Sweep: may_share_memory (across dtypes)', () => {
  describe('may_share_memory', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const a = np.ones([4], dtype);
        expect(np.may_share_memory(a, a)).toBe(true);
      });
    }
  });
});

describe('DType Sweep: shares_memory (across dtypes)', () => {
  describe('shares_memory', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const a = np.ones([4], dtype);
        expect(np.shares_memory(a, a)).toBe(true);
      });
    }
  });
});

describe('DType Sweep: mask_indices (across dtypes)', () => {
  describe('mask_indices', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const pyResult = oracle.get(`mask_indices_${dtype}`)!;
        if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
        const tupleResult = np.mask_indices(3, np.triu);
        const jsResult = np.array(tupleResult.map((a: any) => toComparable(a)));
        expectMatchPre(jsResult, pyResult, { rtol: 0, atol: 0, indexResult: true });
      });
    }
  });
});

describe('DType Sweep: indices (across dtypes)', () => {
  describe('indices', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const pyResult = oracle.get(`indices_${dtype}`)!;
        const r = expectBothRejectPre(
          `indices may not support ${dtype}`,
          () => np.indices([3, 3], dtype),
          pyResult
        );
        if (r === 'both-reject') return;
        const jsResult = np.indices([3, 3], dtype);
        expectMatchPre(jsResult, pyResult, { rtol: 0, atol: 0, indexResult: true });
      });
    }
  });
});

describe('DType Sweep: ravel_multi_index (across dtypes)', () => {
  describe('ravel_multi_index', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const pyResult = oracle.get(`ravel_multi_index_${dtype}`)!;
        const row0 = np.array([1, 0], dtype);
        const row1 = np.array([0, 1], dtype);
        // We intentionally accept float/complex indices (JS Number is always float)
        // NumPy rejects non-integer dtypes, so skip oracle comparison when it errors
        if (pyResult.error) {
          np.ravel_multi_index([row0, row1], [2, 2]); // just verify no throw
          return;
        }
        const jsResult = np.ravel_multi_index([row0, row1], [2, 2]);
        expectMatchPre(jsResult, pyResult, { rtol: 0, atol: 0, indexResult: true });
      });
    }
  });
});

describe('DType Sweep: unravel_index (across dtypes)', () => {
  describe('unravel_index', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const pyResult = oracle.get(`unravel_index_${dtype}`)!;
        const a = np.array([3, 5], dtype);
        // We intentionally accept float/complex indices (JS Number is always float)
        if (pyResult.error) {
          np.unravel_index(a, [3, 3]); // just verify no throw
          return;
        }
        const tupleResult = np.unravel_index(a, [3, 3]);
        const jsResult = np.array(tupleResult.map((x: any) => toComparable(x)));
        expectMatchPre(jsResult, pyResult, { rtol: 0, atol: 0, indexResult: true });
      });
    }
  });
});
