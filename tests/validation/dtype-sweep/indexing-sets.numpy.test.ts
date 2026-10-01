/**
 * DType Sweep: Indexing and set operations.
 * All tested across ALL dtypes, value-producing tests validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectComplexFixture,
  expectMatchPre,
  npDtype,
  pyArrayCast,
  runNumPyBatch,
} from './_helpers';

const { array } = np;

// One definition per fixture, shared by the oracle snippet and the test body:
// the two sides must build the same input, and a separate declaration on each
// side drifts silently. Complex dtypes carry an imaginary part, so a gather,
// scatter or set operation that drops or ignores it shows up as a mismatch.
const fx = {
  take: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1, 0, 1] : [10, 20, 30, 40, 50], d),
  takeAlong: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1] : [10, 20, 30], d),
  put: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1] : [10, 20, 30], d),
  putVals: (d: string) => asDtypeData(d === 'bool' ? [0, 1] : [99, 88], d),
  putAlong: (d: string) =>
    asDtypeData(
      d === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [10, 20],
            [30, 40],
          ],
      d,
    ),
  putAlongVals: (d: string) => asDtypeData(d === 'bool' ? [[1], [0]] : [[99], [88]], d),
  choice0: (d: string) => asDtypeData(d === 'bool' ? [0, 0, 0] : [0, 1, 2], d),
  choice1: (d: string) => asDtypeData(d === 'bool' ? [1, 1, 1] : [10, 11, 12], d),
  diag: (d: string) =>
    asDtypeData(
      d === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ],
      d,
    ),
  whereA: (d: string) => asDtypeData(d === 'bool' ? [1, 1, 1, 1] : [1, 2, 3, 4], d),
  whereB: (d: string) => asDtypeData(d === 'bool' ? [0, 0, 0, 0] : [5, 6, 7, 8], d),
  nonzero: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1] : [0, 1, 0, 2], d),
  extract: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5], d),
  flatnonzero: (d: string) => asDtypeData(d === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0], d),
  argwhere: (d: string) => asDtypeData(d === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0], d),
  setA: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1, 0] : [3, 1, 2, 1, 3, 2], d),
  setB: (d: string) => asDtypeData(d === 'bool' ? [0, 1] : [3, 4, 5, 6], d),
  setC: (d: string) => asDtypeData(d === 'bool' ? [1, 0] : [1, 2, 3, 4], d),
  unionA: (d: string) => asDtypeData(d === 'bool' ? [1, 0] : [1, 2, 3], d),
  unionB: (d: string) => asDtypeData(d === 'bool' ? [0, 1] : [3, 4, 5], d),
  inA: (d: string) => asDtypeData(d === 'bool' ? [1, 0] : [1, 2, 3], d),
  inB: (d: string) => asDtypeData(d === 'bool' ? [1] : [2, 3, 4], d),
};

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);

    // Indexing snippets
    snippets[`take_${dtype}`] =
      `_result_orig = np.take(np.array(${fx.take(dtype).py}, dtype=${npDtype(dtype)}), [0,2,4])
result = _result_orig.astype(${ac})`;

    const takeAlongIndices = [2, 0, 1];
    snippets[`take_along_axis_${dtype}`] = `
a = np.array(${fx.takeAlong(dtype).py}, dtype=${npDtype(dtype)})
idx = np.array(${JSON.stringify(takeAlongIndices)}, dtype=np.intp)
_result_orig = np.take_along_axis(a, idx, axis=0)
result = _result_orig.astype(${ac})`;

    snippets[`put_${dtype}`] = `
a = np.array(${fx.put(dtype).py}, dtype=${npDtype(dtype)})
np.put(a, [0, 2], np.array(${fx.putVals(dtype).py}, dtype=${npDtype(dtype)}))
_result_orig = a
result = _result_orig.astype(${ac})`;

    snippets[`put_along_axis_${dtype}`] = `
a = np.array(${fx.putAlong(dtype).py}, dtype=${npDtype(dtype)})
np.put_along_axis(a, np.array([[0],[1]], dtype=np.intp), np.array(${fx.putAlongVals(dtype).py}, dtype=${npDtype(dtype)}), axis=1)
_result_orig = a
result = _result_orig.astype(${ac})`;

    snippets[`diag_${dtype}`] =
      `_result_orig = np.diag(np.array(${fx.diag(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`where_${dtype}`] = `
cond = np.array([True, False, True, False])
a = np.array(${fx.whereA(dtype).py}, dtype=${npDtype(dtype)})
b = np.array(${fx.whereB(dtype).py}, dtype=${npDtype(dtype)})
_result_orig = np.where(cond, a, b)
result = _result_orig.astype(${ac})`;

    snippets[`nonzero_${dtype}`] =
      `_result_orig = np.nonzero(np.array(${fx.nonzero(dtype).py}, dtype=${npDtype(dtype)}))[0]
result = _result_orig`;

    snippets[`extract_${dtype}`] =
      `_result_orig = np.extract(np.array([True,False,True,False,True]), np.array(${fx.extract(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`flatnonzero_${dtype}`] =
      `_result_orig = np.flatnonzero(np.array(${fx.flatnonzero(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    snippets[`argwhere_${dtype}`] =
      `_result_orig = np.argwhere(np.array(${fx.argwhere(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    // Set operations (skip int64/uint64)
    if (dtype !== 'int64' && dtype !== 'uint64') {
      snippets[`unique_${dtype}`] =
        `_result_orig = np.unique(np.array(${fx.setA(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`intersect1d_${dtype}`] =
        `_result_orig = np.intersect1d(np.array(${fx.setC(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fx.setB(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`union1d_${dtype}`] =
        `_result_orig = np.union1d(np.array(${fx.unionA(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fx.unionB(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`setdiff1d_${dtype}`] =
        `_result_orig = np.setdiff1d(np.array(${fx.setC(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fx.setB(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`setxor1d_${dtype}`] =
        `_result_orig = np.setxor1d(np.array(${fx.setC(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fx.setB(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`in1d_${dtype}`] =
        `_result_orig = np.isin(np.array(${fx.inA(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fx.inB(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`isin_${dtype}`] =
        `_result_orig = np.isin(np.array(${fx.inA(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fx.inB(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;
    }
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Indexing', () => {
  for (const dtype of ALL_DTYPES) {
    it(`take ${dtype}`, () => {
      const a = array(fx.take(dtype).js, dtype);
      expectComplexFixture(a, dtype, `take ${dtype}`);
      const jsResult = np.take(a, [0, 2, 4]);
      expectMatchPre(jsResult, oracle.get(`take_${dtype}`)!);
    });

    it(`take_along_axis ${dtype}`, () => {
      const a = array(fx.takeAlong(dtype).js, dtype);
      expectComplexFixture(a, dtype, `take_along_axis ${dtype}`);
      const jsResult = np.take_along_axis(a, array([2, 0, 1], 'int32'), 0);
      expectMatchPre(jsResult, oracle.get(`take_along_axis_${dtype}`)!);
    });

    it(`put ${dtype}`, () => {
      const a = array(fx.put(dtype).js, dtype);
      expectComplexFixture(a, dtype, `put ${dtype}`);
      np.put(a, [0, 2], array(fx.putVals(dtype).js, dtype));
      expectMatchPre(a, oracle.get(`put_${dtype}`)!);
    });

    it(`put_along_axis ${dtype}`, () => {
      const a = array(fx.putAlong(dtype).js, dtype);
      expectComplexFixture(a, dtype, `put_along_axis ${dtype}`);
      np.put_along_axis(a, array([[0], [1]], 'int32'), array(fx.putAlongVals(dtype).js, dtype), 1);
      expectMatchPre(a, oracle.get(`put_along_axis_${dtype}`)!);
    });

    it(`choose ${dtype}`, () => {
      const choices = [array(fx.choice0(dtype).js, dtype), array(fx.choice1(dtype).js, dtype)];
      expectComplexFixture(choices[0], dtype, `choose ${dtype}`);
      const jsResult = np.choose(array([0, 1, 0], 'int32'), choices);
      expect(jsResult.shape).toEqual([3]);
    });

    it(`diag ${dtype}`, () => {
      const a = array(fx.diag(dtype).js, dtype);
      expectComplexFixture(a, dtype, `diag ${dtype}`);
      const jsResult = np.diag(a);
      expectMatchPre(jsResult, oracle.get(`diag_${dtype}`)!);
    });

    it(`where ${dtype}`, () => {
      const cond = array([1, 0, 1, 0], 'bool');
      const a = array(fx.whereA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `where ${dtype}`);
      const jsResult = np.where(cond, a, array(fx.whereB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`where_${dtype}`)!);
    });

    it(`nonzero ${dtype}`, () => {
      const a = array(fx.nonzero(dtype).js, dtype);
      expectComplexFixture(a, dtype, `nonzero ${dtype}`);
      const jsResult = np.nonzero(a);
      expectMatchPre(jsResult[0]!, oracle.get(`nonzero_${dtype}`)!, {
        indexResult: true,
      });
    });

    it(`extract ${dtype}`, () => {
      const a = array(fx.extract(dtype).js, dtype);
      expectComplexFixture(a, dtype, `extract ${dtype}`);
      const cond = array([1, 0, 1, 0, 1], 'bool');
      const jsResult = np.extract(cond, a);
      expectMatchPre(jsResult, oracle.get(`extract_${dtype}`)!);
    });

    it(`flatnonzero ${dtype}`, () => {
      const a = array(fx.flatnonzero(dtype).js, dtype);
      expectComplexFixture(a, dtype, `flatnonzero ${dtype}`);
      const jsResult = np.flatnonzero(a);
      expectMatchPre(jsResult, oracle.get(`flatnonzero_${dtype}`)!, {
        indexResult: true,
      });
    });

    it(`argwhere ${dtype}`, () => {
      const a = array(fx.argwhere(dtype).js, dtype);
      expectComplexFixture(a, dtype, `argwhere ${dtype}`);
      const jsResult = np.argwhere(a);
      expectMatchPre(jsResult, oracle.get(`argwhere_${dtype}`)!, {
        indexResult: true,
      });
    });
  }
});

describe('DType Sweep: Set operations', () => {
  for (const dtype of ALL_DTYPES) {
    // Skip int64/uint64 — BigInt results can't be compared via arraysClose
    if (dtype === 'int64' || dtype === 'uint64') continue;

    it(`unique ${dtype}`, () => {
      const a = array(fx.setA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `unique ${dtype}`);
      expectMatchPre(np.unique(a), oracle.get(`unique_${dtype}`)!);
    });

    it(`intersect1d ${dtype}`, () => {
      const a = array(fx.setC(dtype).js, dtype);
      expectComplexFixture(a, dtype, `intersect1d ${dtype}`);
      const jsResult = np.intersect1d(a, array(fx.setB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`intersect1d_${dtype}`)!);
    });

    it(`union1d ${dtype}`, () => {
      const a = array(fx.unionA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `union1d ${dtype}`);
      const jsResult = np.union1d(a, array(fx.unionB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`union1d_${dtype}`)!);
    });

    it(`setdiff1d ${dtype}`, () => {
      const a = array(fx.setC(dtype).js, dtype);
      expectComplexFixture(a, dtype, `setdiff1d ${dtype}`);
      const jsResult = np.setdiff1d(a, array(fx.setB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`setdiff1d_${dtype}`)!);
    });

    it(`setxor1d ${dtype}`, () => {
      const a = array(fx.setC(dtype).js, dtype);
      expectComplexFixture(a, dtype, `setxor1d ${dtype}`);
      const jsResult = np.setxor1d(a, array(fx.setB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`setxor1d_${dtype}`)!);
    });

    it(`in1d ${dtype}`, () => {
      const a = array(fx.inA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `in1d ${dtype}`);
      const jsResult = np.in1d(a, array(fx.inB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`in1d_${dtype}`)!);
    });

    it(`isin ${dtype}`, () => {
      const a = array(fx.inA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `isin ${dtype}`);
      const jsResult = np.isin(a, array(fx.inB(dtype).js, dtype));
      expectMatchPre(jsResult, oracle.get(`isin_${dtype}`)!);
    });
  }
});
