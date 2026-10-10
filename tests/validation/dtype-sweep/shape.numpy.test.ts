/**
 * DType Sweep: Shape manipulation functions.
 * Structural tests — validates shape/dtype preservation across ALL dtypes.
 * Oracle tests use batched oracle for roll, flip, rot90.
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

const SMALL_DATA = [1, 2, 3, 4, 5, 6];
const SMALL_2D = [
  [1, 2, 3],
  [4, 5, 6],
];

// Each fixture has exactly one definition, shared by the oracle snippets and
// the test bodies. Two definitions of the same fixture drift silently once a
// dtype carries an imaginary part that only one side builds.
const data6 = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : SMALL_DATA, dtype);
const data3 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
const data4 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
const data2dFor = (dtype: string) =>
  asDtypeData(
    dtype === 'bool'
      ? [
          [1, 0, 1],
          [0, 1, 0],
        ]
      : SMALL_2D,
    dtype,
  );
const pairA = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
const pairB = (dtype: string) => asDtypeData(dtype === 'bool' ? [0, 1] : [3, 4], dtype);
const single = (dtype: string) => asDtypeData([1], dtype);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const d = data3(dtype);
    const data2d = data2dFor(dtype);

    snippets[`shape_roll_${dtype}`] =
      `_result_orig = np.roll(np.array(${d.py}, dtype=${npDtype(dtype)}), 1)
result = _result_orig.astype(${ac})`;

    snippets[`shape_flip_${dtype}`] =
      `_result_orig = np.flip(np.array(${d.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`shape_rot90_${dtype}`] =
      `_result_orig = np.rot90(np.array(${data2d.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Shape manipulation', () => {
  for (const dtype of ALL_DTYPES) {
    it(`reshape ${dtype}`, () => {
      const a = array(data6(dtype).js, dtype);
      expectComplexFixture(a, dtype, `reshape ${dtype}`);
      const r = np.reshape(a, [2, 3]);
      expect(r.shape).toEqual([2, 3]);
      expectComplexFixture(r, dtype, `reshape ${dtype} result`);
    });

    it(`transpose ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `transpose ${dtype}`);
      const r = np.transpose(a);
      expect(r.shape).toEqual([3, 2]);
      expectComplexFixture(r, dtype, `transpose ${dtype} result`);
    });

    it(`ravel ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `ravel ${dtype}`);
      const r = np.ravel(a);
      expect(r.shape).toEqual([6]);
      expectComplexFixture(r, dtype, `ravel ${dtype} result`);
    });

    it(`concatenate ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      const b = array(pairB(dtype).js, dtype);
      expectComplexFixture(a, dtype, `concatenate ${dtype}`);
      const r = np.concatenate([a, b]);
      expect(r.shape).toEqual([4]);
      expect(r.dtype).toBe(dtype);
      expectComplexFixture(r, dtype, `concatenate ${dtype} result`);
    });

    it(`stack ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      const b = array(pairB(dtype).js, dtype);
      expectComplexFixture(a, dtype, `stack ${dtype}`);
      const r = np.stack([a, b]);
      expect(r.shape).toEqual([2, 2]);
      expectComplexFixture(r, dtype, `stack ${dtype} result`);
    });

    it(`tile ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `tile ${dtype}`);
      const r = np.tile(a, 3);
      expect(r.shape).toEqual([6]);
      expectComplexFixture(r, dtype, `tile ${dtype} result`);
    });

    it(`repeat ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `repeat ${dtype}`);
      const r = np.repeat(a, 2);
      expect(r.shape).toEqual([4]);
      expectComplexFixture(r, dtype, `repeat ${dtype} result`);
    });

    it(`roll ${dtype}`, () => {
      const d = data3(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `roll ${dtype}`);
      const jsResult = np.roll(a, 1);
      expectMatchPre(jsResult, oracle.get(`shape_roll_${dtype}`)!);
    });

    it(`flip ${dtype}`, () => {
      const d = data3(dtype);
      const a = array(d.js, dtype);
      expectComplexFixture(a, dtype, `flip ${dtype}`);
      const jsResult = np.flip(a);
      expectMatchPre(jsResult, oracle.get(`shape_flip_${dtype}`)!);
    });

    it(`squeeze ${dtype}`, () => {
      const a = array(asDtypeData(dtype === 'bool' ? [[1, 0]] : [[1, 2]], dtype).js, dtype);
      expectComplexFixture(a, dtype, `squeeze ${dtype}`);
      const r = np.squeeze(a);
      expect(r.shape).toEqual([2]);
      expectComplexFixture(r, dtype, `squeeze ${dtype} result`);
    });

    it(`expand_dims ${dtype}`, () => {
      const a = array(single(dtype).js, dtype);
      expectComplexFixture(a, dtype, `expand_dims ${dtype}`);
      const r = np.expand_dims(a, 0);
      expect(r.shape).toEqual([1, 1]);
      expectComplexFixture(r, dtype, `expand_dims ${dtype} result`);
    });

    it(`hstack ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `hstack ${dtype}`);
      const r = np.hstack([a, a]);
      expect(r.shape).toEqual([4]);
      expectComplexFixture(r, dtype, `hstack ${dtype} result`);
    });

    it(`vstack ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `vstack ${dtype}`);
      const r = np.vstack([a, a]);
      expect(r.shape).toEqual([2, 2]);
      expectComplexFixture(r, dtype, `vstack ${dtype} result`);
    });

    it(`dstack ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `dstack ${dtype}`);
      const r = np.dstack([a, a]);
      expect(r.shape).toEqual([1, 2, 2]);
      expectComplexFixture(r, dtype, `dstack ${dtype} result`);
    });

    it(`column_stack ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `column_stack ${dtype}`);
      const r = np.column_stack([a, a]);
      expect(r.shape).toEqual([2, 2]);
      expectComplexFixture(r, dtype, `column_stack ${dtype} result`);
    });

    it(`append ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `append ${dtype}`);
      const r = np.append(a, a);
      expect(r.shape).toEqual([4]);
      expectComplexFixture(r, dtype, `append ${dtype} result`);
    });

    it(`atleast_1d ${dtype}`, () => {
      const a = array(single(dtype).js, dtype);
      expectComplexFixture(a, dtype, `atleast_1d ${dtype}`);
      const r = np.atleast_1d(a);
      expect(r.ndim).toBeGreaterThanOrEqual(1);
      expectComplexFixture(r, dtype, `atleast_1d ${dtype} result`);
    });

    it(`atleast_2d ${dtype}`, () => {
      const a = array(single(dtype).js, dtype);
      expectComplexFixture(a, dtype, `atleast_2d ${dtype}`);
      const r = np.atleast_2d(a);
      expect(r.ndim).toBeGreaterThanOrEqual(2);
      expectComplexFixture(r, dtype, `atleast_2d ${dtype} result`);
    });

    it(`atleast_3d ${dtype}`, () => {
      const a = array(single(dtype).js, dtype);
      expectComplexFixture(a, dtype, `atleast_3d ${dtype}`);
      const r = np.atleast_3d(a);
      expect(r.ndim).toBeGreaterThanOrEqual(3);
      expectComplexFixture(r, dtype, `atleast_3d ${dtype} result`);
    });

    it(`broadcast_to ${dtype}`, () => {
      const a = array(single(dtype).js, dtype);
      expectComplexFixture(a, dtype, `broadcast_to ${dtype}`);
      const r = np.broadcast_to(a, [3]);
      expect(r.shape).toEqual([3]);
      expectComplexFixture(r, dtype, `broadcast_to ${dtype} result`);
    });

    it(`broadcast_arrays ${dtype}`, () => {
      const a = array(single(dtype).js, dtype);
      const b = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `broadcast_arrays ${dtype}`);
      const [ra, rb] = np.broadcast_arrays(a, b) as any[];
      expect(ra.shape).toEqual([3]);
      expect(rb.shape).toEqual([3]);
      expectComplexFixture(ra, dtype, `broadcast_arrays ${dtype} result a`);
      expectComplexFixture(rb, dtype, `broadcast_arrays ${dtype} result b`);
    });

    it(`split ${dtype}`, () => {
      const a = array(data4(dtype).js, dtype);
      expectComplexFixture(a, dtype, `split ${dtype}`);
      const parts = np.split(a, 2);
      expect(parts.length).toBe(2);
      expectComplexFixture(parts[0], dtype, `split ${dtype} result`);
    });

    it(`hsplit ${dtype}`, () => {
      const a = array(data4(dtype).js, dtype);
      expectComplexFixture(a, dtype, `hsplit ${dtype}`);
      const parts = np.hsplit(a, 2);
      expect(parts.length).toBe(2);
      expectComplexFixture(parts[0], dtype, `hsplit ${dtype} result`);
    });

    it(`vsplit ${dtype}`, () => {
      const a = array(
        asDtypeData(
          dtype === 'bool'
            ? [
                [1, 0],
                [0, 1],
              ]
            : [
                [1, 2],
                [3, 4],
              ],
          dtype,
        ).js,
        dtype,
      );
      expectComplexFixture(a, dtype, `vsplit ${dtype}`);
      const parts = np.vsplit(a, 2);
      expect(parts.length).toBe(2);
      expectComplexFixture(parts[0], dtype, `vsplit ${dtype} result`);
    });

    it(`dsplit ${dtype}`, () => {
      const flat = asDtypeData(
        dtype === 'bool' ? [1, 0, 1, 0, 1, 0, 1, 0] : [1, 2, 3, 4, 5, 6, 7, 8],
        dtype,
      );
      const a = np.reshape(array(flat.js, dtype), [2, 2, 2]);
      expectComplexFixture(a, dtype, `dsplit ${dtype}`);
      const parts = np.dsplit(a, 2);
      expect(parts.length).toBe(2);
      expectComplexFixture(parts[0], dtype, `dsplit ${dtype} result`);
    });

    it(`array_split ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `array_split ${dtype}`);
      const parts = np.array_split(a, 2);
      expect(parts.length).toBe(2);
      expectComplexFixture(parts[0], dtype, `array_split ${dtype} result`);
    });

    it(`fliplr ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `fliplr ${dtype}`);
      const r = np.fliplr(a);
      expect(r.shape).toEqual([2, 3]);
      expectComplexFixture(r, dtype, `fliplr ${dtype} result`);
    });

    it(`flipud ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `flipud ${dtype}`);
      const r = np.flipud(a);
      expect(r.shape).toEqual([2, 3]);
      expectComplexFixture(r, dtype, `flipud ${dtype} result`);
    });

    it(`rot90 ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `rot90 ${dtype}`);
      const jsResult = np.rot90(a);
      expectMatchPre(jsResult, oracle.get(`shape_rot90_${dtype}`)!);
    });

    it(`moveaxis ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `moveaxis ${dtype}`);
      const r = np.moveaxis(a, 0, 1);
      expect(r.shape).toEqual([3, 2]);
      expectComplexFixture(r, dtype, `moveaxis ${dtype} result`);
    });

    it(`swapaxes ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `swapaxes ${dtype}`);
      const r = np.swapaxes(a, 0, 1);
      expect(r.shape).toEqual([3, 2]);
      expectComplexFixture(r, dtype, `swapaxes ${dtype} result`);
    });

    it(`insert ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `insert ${dtype}`);
      const jsResult = np.insert(a, 1, dtype === 'bool' ? 0 : 99);
      expect(jsResult.shape).toEqual([4]);
      expectComplexFixture(jsResult, dtype, `insert ${dtype} result`);
    });

    it(`delete_ ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `delete_ ${dtype}`);
      const r = np.delete_(a, 1);
      expect(r.shape).toEqual([2]);
      expectComplexFixture(r, dtype, `delete_ ${dtype} result`);
    });

    it(`resize ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `resize ${dtype}`);
      const r = np.resize(a, [4]);
      expect(r.shape).toEqual([4]);
      expectComplexFixture(r, dtype, `resize ${dtype} result`);
    });

    it(`unstack ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `unstack ${dtype}`);
      const result = np.unstack(a);
      expect(result.length).toBe(2);
      expect(result[0]!.shape).toEqual([3]);
      expectComplexFixture(result[0], dtype, `unstack ${dtype} result`);
    });

    it(`diagflat ${dtype}`, () => {
      const a = array(pairA(dtype).js, dtype);
      expectComplexFixture(a, dtype, `diagflat ${dtype}`);
      const jsResult = np.diagflat(a);
      expect(jsResult.shape).toEqual([2, 2]);
      expectComplexFixture(jsResult, dtype, `diagflat ${dtype} result`);
    });

    it(`flatten ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `flatten ${dtype}`);
      const r = np.flatten(a);
      expect(r.shape).toEqual([6]);
      expectComplexFixture(r, dtype, `flatten ${dtype} result`);
    });

    it(`pad ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `pad ${dtype}`);
      const r = np.pad(a, 2);
      expect(r.shape).toEqual([7]);
    });

    it(`trim_zeros ${dtype}`, () => {
      // Left real for every dtype: trim_zeros keys off exact zeros, and a
      // complex fixture would give the padding zeros an imaginary part.
      const data0 = dtype === 'bool' ? [0, 1, 0] : [0, 1, 2, 0];
      const jsResult = np.trim_zeros(array(data0, dtype));
      const expected = dtype === 'bool' ? [1] : [1, 2];
      expect(jsResult.shape).toEqual([expected.length]);
    });

    it(`compress ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `compress ${dtype}`);
      const cond = array([1, 0, 1], 'bool');
      const r = np.compress(cond, a);
      expect(r.shape).toEqual([2]);
      expectComplexFixture(r, dtype, `compress ${dtype} result`);
    });

    it(`select ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `select ${dtype}`);
      const cond = array([1, 0, 1], 'bool');
      const result = np.select([cond], [a]);
      expect(result.shape).toEqual([3]);
      expectComplexFixture(result, dtype, `select ${dtype} result`);
    });

    it(`diag ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `diag ${dtype}`);
      const r = np.diag(a);
      expect(r.shape).toEqual([2]);
      expectComplexFixture(r, dtype, `diag ${dtype} result`);
    });

    it(`diagonal ${dtype}`, () => {
      const a = array(data2dFor(dtype).js, dtype);
      expectComplexFixture(a, dtype, `diagonal ${dtype}`);
      const r = np.diagonal(a);
      expect(r.shape).toEqual([2]);
      expectComplexFixture(r, dtype, `diagonal ${dtype} result`);
    });

    it(`fill_diagonal ${dtype}`, () => {
      const a = array(
        asDtypeData(
          [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
          dtype,
        ).js,
        dtype,
      );
      expectComplexFixture(a, dtype, `fill_diagonal ${dtype}`);
      np.fill_diagonal(a, dtype === 'bool' ? 1 : 9);
      expect(a.shape).toEqual([3, 3]);
    });
  }
});
