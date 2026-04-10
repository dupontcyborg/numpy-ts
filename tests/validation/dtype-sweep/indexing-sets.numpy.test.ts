/**
 * DType Sweep: Indexing and set operations.
 * All tested across ALL dtypes, value-producing tests validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  pyArrayCast,
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
    const ac = pyArrayCast(dtype);

    // Indexing snippets
    const takeData = dtype === 'bool' ? [1, 0, 1, 0, 1] : [10, 20, 30, 40, 50];
    snippets[`take_${dtype}`] =
      `_result_orig = np.take(np.array(${JSON.stringify(takeData)}, dtype=${npDtype(dtype)}), [0,2,4])
result = _result_orig.astype(${ac})`;

    const takeAlongData = dtype === 'bool' ? [1, 0, 1] : [10, 20, 30];
    const takeAlongIndices = [2, 0, 1];
    snippets[`take_along_axis_${dtype}`] = `
a = np.array(${JSON.stringify(takeAlongData)}, dtype=${npDtype(dtype)})
idx = np.array(${JSON.stringify(takeAlongIndices)}, dtype=np.intp)
_result_orig = np.take_along_axis(a, idx, axis=0)
result = _result_orig.astype(${ac})`;

    const putData = dtype === 'bool' ? [1, 0, 1] : [10, 20, 30];
    const putVals = dtype === 'bool' ? [0, 1] : [99, 88];
    snippets[`put_${dtype}`] = `
a = np.array(${JSON.stringify(putData)}, dtype=${npDtype(dtype)})
np.put(a, [0, 2], np.array(${JSON.stringify(putVals)}, dtype=${npDtype(dtype)}))
_result_orig = a
result = _result_orig.astype(${ac})`;

    const putAlongData =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [10, 20],
            [30, 40],
          ];
    const putAlongVals = dtype === 'bool' ? [[1], [0]] : [[99], [88]];
    snippets[`put_along_axis_${dtype}`] = `
a = np.array(${JSON.stringify(putAlongData)}, dtype=${npDtype(dtype)})
np.put_along_axis(a, np.array([[0],[1]], dtype=np.intp), np.array(${JSON.stringify(putAlongVals)}, dtype=${npDtype(dtype)}), axis=1)
_result_orig = a
result = _result_orig.astype(${ac})`;

    const diagData = dtype === 'bool' ? '[[True,False],[False,True]]' : '[[1,2],[3,4]]';
    snippets[`diag_${dtype}`] =
      `_result_orig = np.diag(np.array(${diagData}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    const data1 = dtype === 'bool' ? [1, 1, 1, 1] : [1, 2, 3, 4];
    const data2 = dtype === 'bool' ? [0, 0, 0, 0] : [5, 6, 7, 8];
    snippets[`where_${dtype}`] = `
cond = np.array([True, False, True, False])
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
_result_orig = np.where(cond, a, b)
result = _result_orig.astype(${ac})`;

    const nonzeroData = dtype === 'bool' ? [1, 0, 1] : [0, 1, 0, 2];
    snippets[`nonzero_${dtype}`] =
      `_result_orig = np.nonzero(np.array(${JSON.stringify(nonzeroData)}, dtype=${npDtype(dtype)}))[0]
result = _result_orig`;

    const extractData = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
    snippets[`extract_${dtype}`] =
      `_result_orig = np.extract(np.array([True,False,True,False,True]), np.array(${JSON.stringify(extractData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    const flatnonzeroData = dtype === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0];
    snippets[`flatnonzero_${dtype}`] =
      `_result_orig = np.flatnonzero(np.array(${JSON.stringify(flatnonzeroData)}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    const argwhereData = dtype === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0];
    snippets[`argwhere_${dtype}`] =
      `_result_orig = np.argwhere(np.array(${JSON.stringify(argwhereData)}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    // Set operations (skip int64/uint64)
    if (dtype !== 'int64' && dtype !== 'uint64') {
      const d1 = dtype === 'bool' ? [1, 0, 1, 0] : [3, 1, 2, 1, 3, 2];
      const d2 = dtype === 'bool' ? [0, 1] : [3, 4, 5, 6];
      const d3 = dtype === 'bool' ? [1, 0] : [1, 2, 3, 4];

      snippets[`unique_${dtype}`] =
        `_result_orig = np.unique(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`intersect1d_${dtype}`] =
        `_result_orig = np.intersect1d(np.array(${JSON.stringify(d3)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      const union1 = dtype === 'bool' ? [1, 0] : [1, 2, 3];
      const union2 = dtype === 'bool' ? [0, 1] : [3, 4, 5];
      snippets[`union1d_${dtype}`] =
        `_result_orig = np.union1d(np.array(${JSON.stringify(union1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(union2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`setdiff1d_${dtype}`] =
        `_result_orig = np.setdiff1d(np.array(${JSON.stringify(d3)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`setxor1d_${dtype}`] =
        `_result_orig = np.setxor1d(np.array(${JSON.stringify(d3)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      const in1d1 = dtype === 'bool' ? [1, 0] : [1, 2, 3];
      const in1d2 = dtype === 'bool' ? [1] : [2, 3, 4];
      snippets[`in1d_${dtype}`] =
        `_result_orig = np.in1d(np.array(${JSON.stringify(in1d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(in1d2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

      snippets[`isin_${dtype}`] =
        `_result_orig = np.isin(np.array(${JSON.stringify(in1d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(in1d2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;
    }
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Indexing', () => {
  for (const dtype of ALL_DTYPES) {
    it(`take ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [10, 20, 30, 40, 50];
      const jsResult = np.take(array(data, dtype), [0, 2, 4]);
      expectMatchPre(jsResult, oracle.get(`take_${dtype}`)!);
    });

    it(`take_along_axis ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [10, 20, 30];
      const indices = [2, 0, 1];
      const jsResult = np.take_along_axis(array(data, dtype), array(indices, 'int32'), 0);
      expectMatchPre(jsResult, oracle.get(`take_along_axis_${dtype}`)!);
    });

    it(`put ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [10, 20, 30];
      const vals = dtype === 'bool' ? [0, 1] : [99, 88];
      const a = array([...data], dtype);
      np.put(a, [0, 2], array(vals, dtype));
      expectMatchPre(a, oracle.get(`put_${dtype}`)!);
    });

    it(`put_along_axis ${dtype}`, () => {
      const data =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [10, 20],
              [30, 40],
            ];
      const vals = dtype === 'bool' ? [[1], [0]] : [[99], [88]];
      const a = array(data, dtype);
      np.put_along_axis(a, array([[0], [1]], 'int32'), array(vals, dtype), 1);
      expectMatchPre(a, oracle.get(`put_along_axis_${dtype}`)!);
    });

    it(`choose ${dtype}`, () => {
      const choices = [
        array(dtype === 'bool' ? [0, 0, 0] : [0, 1, 2], dtype),
        array(dtype === 'bool' ? [1, 1, 1] : [10, 11, 12], dtype),
      ];
      const jsResult = np.choose(array([0, 1, 0], 'int32'), choices);
      expect(jsResult.shape).toEqual([3]);
    });

    it(`diag ${dtype}`, () => {
      const jsResult = np.diag(
        array(
          dtype === 'bool'
            ? [
                [1, 0],
                [0, 1],
              ]
            : [
                [1, 2],
                [3, 4],
              ],
          dtype
        )
      );
      expectMatchPre(jsResult, oracle.get(`diag_${dtype}`)!);
    });

    it(`where ${dtype}`, () => {
      const cond = array([1, 0, 1, 0], 'bool');
      const data1 = dtype === 'bool' ? [1, 1, 1, 1] : [1, 2, 3, 4];
      const data2 = dtype === 'bool' ? [0, 0, 0, 0] : [5, 6, 7, 8];
      const jsResult = np.where(cond, array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, oracle.get(`where_${dtype}`)!);
    });

    it(`nonzero ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [0, 1, 0, 2];
      const jsResult = np.nonzero(array(data, dtype));
      expectMatchPre(jsResult[0]!, oracle.get(`nonzero_${dtype}`)!, { indexResult: true });
    });

    it(`extract ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
      const a = array(data, dtype);
      const cond = array([1, 0, 1, 0, 1], 'bool');
      const jsResult = np.extract(cond, a);
      expectMatchPre(jsResult, oracle.get(`extract_${dtype}`)!);
    });

    it(`flatnonzero ${dtype}`, () => {
      const data = dtype === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0];
      const jsResult = np.flatnonzero(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`flatnonzero_${dtype}`)!, { indexResult: true });
    });

    it(`argwhere ${dtype}`, () => {
      const data = dtype === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0];
      const jsResult = np.argwhere(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`argwhere_${dtype}`)!, { indexResult: true });
    });
  }
});

describe('DType Sweep: Set operations', () => {
  for (const dtype of ALL_DTYPES) {
    // Skip int64/uint64 — BigInt results can't be compared via arraysClose
    if (dtype === 'int64' || dtype === 'uint64') continue;

    const d1 = dtype === 'bool' ? [1, 0, 1, 0] : [3, 1, 2, 1, 3, 2];
    const d2 = dtype === 'bool' ? [0, 1] : [3, 4, 5, 6];
    const d3 = dtype === 'bool' ? [1, 0] : [1, 2, 3, 4];

    it(`unique ${dtype}`, () => {
      const jsResult = np.unique(array(d1, dtype));
      expectMatchPre(jsResult, oracle.get(`unique_${dtype}`)!);
    });

    it(`intersect1d ${dtype}`, () => {
      const jsResult = np.intersect1d(array(d3, dtype), array(d2, dtype));
      expectMatchPre(jsResult, oracle.get(`intersect1d_${dtype}`)!);
    });

    it(`union1d ${dtype}`, () => {
      const jsResult = np.union1d(
        array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype),
        array(dtype === 'bool' ? [0, 1] : [3, 4, 5], dtype)
      );
      expectMatchPre(jsResult, oracle.get(`union1d_${dtype}`)!);
    });

    it(`setdiff1d ${dtype}`, () => {
      const jsResult = np.setdiff1d(array(d3, dtype), array(d2, dtype));
      expectMatchPre(jsResult, oracle.get(`setdiff1d_${dtype}`)!);
    });

    it(`setxor1d ${dtype}`, () => {
      const jsResult = np.setxor1d(array(d3, dtype), array(d2, dtype));
      expectMatchPre(jsResult, oracle.get(`setxor1d_${dtype}`)!);
    });

    it(`in1d ${dtype}`, () => {
      const jsResult = np.in1d(
        array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype),
        array(dtype === 'bool' ? [1] : [2, 3, 4], dtype)
      );
      expectMatchPre(jsResult, oracle.get(`in1d_${dtype}`)!);
    });

    it(`isin ${dtype}`, () => {
      const jsResult = np.isin(
        array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype),
        array(dtype === 'bool' ? [1] : [2, 3, 4], dtype)
      );
      expectMatchPre(jsResult, oracle.get(`isin_${dtype}`)!);
    });
  }
});
