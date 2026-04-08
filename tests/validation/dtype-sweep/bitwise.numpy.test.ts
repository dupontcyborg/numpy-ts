/**
 * DType Sweep: Bitwise operations + packbits/unpackbits, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  checkNumPyAvailable,
  npDtype,
  runNumPyBatch,
  expectBothRejectPre,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

const BITWISE_UNSUPPORTED = (d: string) => d.startsWith('float') || d.startsWith('complex');

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : [15, 7, 3, 1];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [9, 6, 5, 3];

    snippets[`bitwise_and_${dtype}`] =
      `_result_orig = np.bitwise_and(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;
    snippets[`bitwise_or_${dtype}`] =
      `_result_orig = np.bitwise_or(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;
    snippets[`bitwise_xor_${dtype}`] =
      `_result_orig = np.bitwise_xor(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;
    snippets[`bitwise_not_${dtype}`] =
      `_result_orig = np.bitwise_not(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;
    snippets[`invert_${dtype}`] =
      `_result_orig = np.invert(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;

    const d1s = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    const d2s = dtype === 'bool' ? [1, 1, 0] : [1, 1, 1];
    snippets[`left_shift_${dtype}`] =
      `_result_orig = np.left_shift(np.array(${JSON.stringify(d1s)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2s)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;

    const d1r = dtype === 'bool' ? [1, 0, 1] : [8, 16, 32];
    const d2r = dtype === 'bool' ? [1, 0, 0] : [1, 1, 1];
    snippets[`right_shift_${dtype}`] =
      `_result_orig = np.right_shift(np.array(${JSON.stringify(d1r)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2r)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;

    const countData = dtype === 'bool' ? [1, 0, 1] : [7, 15, 3];
    snippets[`bitwise_count_${dtype}`] =
      `_result_orig = np.bitwise_count(np.array(${JSON.stringify(countData)}, dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;

    // Packbits/unpackbits
    snippets[`packbits_${dtype}`] =
      `_result_orig = np.packbits(np.array([1,0,1,0,1,0,1,0], dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;
    snippets[`unpackbits_${dtype}`] =
      `_result_orig = np.unpackbits(np.array([170], dtype=${npDtype(dtype)}))\nresult = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Bitwise', () => {
  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : [15, 7, 3, 1];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [9, 6, 5, 3];

    it(`bitwise_and ${dtype}`, () => {
      const py = oracle.get(`bitwise_and_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'bitwise_and requires integer or bool dtype',
          () => np.bitwise_and(array(data1, dtype), array(data2, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.bitwise_and(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`bitwise_or ${dtype}`, () => {
      const py = oracle.get(`bitwise_or_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'bitwise_or requires integer or bool dtype',
          () => np.bitwise_or(array(data1, dtype), array(data2, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.bitwise_or(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`bitwise_xor ${dtype}`, () => {
      const py = oracle.get(`bitwise_xor_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'bitwise_xor requires integer or bool dtype',
          () => np.bitwise_xor(array(data1, dtype), array(data2, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.bitwise_xor(array(data1, dtype), array(data2, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`bitwise_not ${dtype}`, () => {
      const py = oracle.get(`bitwise_not_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'bitwise_not requires integer or bool dtype',
          () => np.bitwise_not(array(data1, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.bitwise_not(array(data1, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`invert ${dtype}`, () => {
      const py = oracle.get(`invert_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'invert requires integer or bool dtype',
          () => np.invert(array(data1, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.invert(array(data1, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`left_shift ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [1, 1, 0] : [1, 1, 1];
      const py = oracle.get(`left_shift_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'left_shift requires integer or bool dtype',
          () => np.left_shift(array(d1, dtype), array(d2, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.left_shift(array(d1, dtype), array(d2, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`right_shift ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [8, 16, 32];
      const d2 = dtype === 'bool' ? [1, 0, 0] : [1, 1, 1];
      const py = oracle.get(`right_shift_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'right_shift requires integer or bool dtype',
          () => np.right_shift(array(d1, dtype), array(d2, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.right_shift(array(d1, dtype), array(d2, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`bitwise_count ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [7, 15, 3];
      const py = oracle.get(`bitwise_count_${dtype}`)!;
      if (BITWISE_UNSUPPORTED(dtype)) {
        const _r = expectBothRejectPre(
          'bitwise_count requires integer or bool dtype',
          () => np.bitwise_count(array(data, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.bitwise_count(array(data, dtype));
      expectMatchPre(jsResult, py);
    });
  }
});

describe('DType Sweep: Packbits/Unpackbits', () => {
  for (const dtype of ALL_DTYPES) {
    it(`packbits ${dtype}`, () => {
      const data = [1, 0, 1, 0, 1, 0, 1, 0];
      const py = oracle.get(`packbits_${dtype}`)!;
      if (dtype !== 'uint8') {
        const _r = expectBothRejectPre(
          'packbits only accepts uint8 arrays',
          () => np.packbits(array(data, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.packbits(array(data, dtype));
      expectMatchPre(jsResult, py);
    });

    it(`unpackbits ${dtype}`, () => {
      const data = [170];
      const py = oracle.get(`unpackbits_${dtype}`)!;
      if (dtype !== 'uint8') {
        const _r = expectBothRejectPre(
          'unpackbits only accepts uint8 arrays',
          () => np.unpackbits(array(data, dtype)),
          py
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.unpackbits(array(data, dtype));
      expectMatchPre(jsResult, py);
    });
  }
});
